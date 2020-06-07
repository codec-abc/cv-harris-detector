use cv::nalgebra::Point3;
use cv::{
    camera::pinhole::{CameraIntrinsicsK1Distortion, NormalizedKeyPoint},
    feature::akaze,
    Bearing, BitArray, CameraModel, Consensus, EssentialMatrix, Estimator, FeatureMatch,
    FeatureWorldMatch, Pose, RelativeCameraPose, TriangulatorObservances, TriangulatorRelative,
    WorldPoint, WorldPose,
};
use cv_optimize::{ManyViewOptimizer, TwoViewOptimizer};
use image::DynamicImage;
use levenberg_marquardt::LevenbergMarquardt;
use log::*;
use rand::{seq::SliceRandom, Rng};
use slab::Slab;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

type Features = Vec<(NormalizedKeyPoint, BitArray<64>)>;

struct Frame {
    /// A VSlam::feeds index
    feed: usize,
    /// The keypoints and corresponding descriptors observed on this frame
    features: Features,
    /// A map from Frame::features indices to VSlam::landmarks indices
    landmarks: HashMap<usize, usize>,
    /// A list of outgoing (first in pair) covisibilities
    covisibilities: HashSet<Pair>,
}

impl Frame {
    fn keypoints(&self) -> impl Iterator<Item = NormalizedKeyPoint> + Clone + '_ {
        self.features.iter().map(|&(kp, _)| kp)
    }

    fn descriptors(&self) -> impl Iterator<Item = BitArray<64>> + Clone + '_ {
        self.features.iter().map(|&(_, d)| d)
    }

    fn keypoint(&self, ix: usize) -> NormalizedKeyPoint {
        self.features[ix].0
    }

    fn descriptor(&self, ix: usize) -> BitArray<64> {
        self.features[ix].1
    }
}

/// A 3d point in space that has been observed on two or more frames
struct Landmark {
    /// Contains a map from VSlam::frames indices to Frame::features indices.
    observances: HashMap<usize, usize>,
}

struct Feed {
    /// The camera intrinsics for this feed
    intrinsics: CameraIntrinsicsK1Distortion,
    /// VSlam::frames indices corresponding to each frame of the feed
    frames: Vec<usize>,
}

struct Covisibility {
    /// The relative pose between the first to the second frame in the covisibility
    pose: RelativeCameraPose,
    /// The matches in index from the first view to the second
    matches: Vec<FeatureMatch<usize>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair(usize, usize);

impl Pair {
    /// Creates a new pair, cannonicalizing the order of the pair.
    pub fn new(a: usize, b: usize) -> Self {
        Self(std::cmp::min(a, b), std::cmp::max(a, b))
    }
}

pub struct BundleAdjust {
    poses: HashMap<usize, WorldPose>,
    pairs: Vec<Pair>,
}

pub struct VSlam<C, EE, PE, T, R> {
    /// Contains the camera intrinsics for each feed
    feeds: Slab<Feed>,
    /// Contains all the landmarks
    landmarks: Slab<Landmark>,
    /// Contains all the frames
    frames: Slab<Frame>,
    /// Contains a mapping from cannonical pairs to covisibility data
    covisibilities: HashMap<Pair, Option<Covisibility>>,
    /// The threshold used for akaze
    akaze_threshold: f64,
    /// The threshold distance below which a match is allowed
    match_threshold: usize,
    /// The number of points to use in optimization of matches
    optimization_points: usize,
    /// The maximum cosine distance permitted in a valid match
    cosine_distance_threshold: f64,
    /// The consensus algorithm
    consensus: RefCell<C>,
    /// The essential matrix estimator
    essential_estimator: EE,
    /// The PnP estimator
    pose_estimator: PE,
    /// The triangulation algorithm
    triangulator: T,
    /// The random number generator
    rng: RefCell<R>,
}

impl<C, EE, PE, T, R> VSlam<C, EE, PE, T, R>
where
    C: Consensus<EE, FeatureMatch<NormalizedKeyPoint>>,
    EE: Estimator<FeatureMatch<NormalizedKeyPoint>, Model = EssentialMatrix>,
    PE: Estimator<FeatureWorldMatch<NormalizedKeyPoint>, Model = WorldPose>,
    T: TriangulatorObservances + Clone,
    R: Rng,
{
    /// Creates an empty vSLAM reconstruction.
    pub fn new(
        consensus: C,
        essential_estimator: EE,
        pose_estimator: PE,
        triangulator: T,
        rng: R,
    ) -> Self {
        Self {
            feeds: Default::default(),
            landmarks: Default::default(),
            frames: Default::default(),
            covisibilities: Default::default(),
            akaze_threshold: 0.001,
            match_threshold: 64,
            cosine_distance_threshold: 0.001,
            optimization_points: 16,
            consensus: RefCell::new(consensus),
            essential_estimator,
            pose_estimator,
            triangulator,
            rng: RefCell::new(rng),
        }
    }

    /// Set the akaze threshold.
    ///
    /// Default: `0.001`
    pub fn akaze_threshold(self, akaze_threshold: f64) -> Self {
        Self {
            akaze_threshold,
            ..self
        }
    }

    /// Set the match threshold.
    ///
    /// Default: `64`
    pub fn match_threshold(self, match_threshold: usize) -> Self {
        Self {
            match_threshold,
            ..self
        }
    }

    /// Set the number of points used for optimization of matching.
    ///
    /// Default: `16`
    pub fn optimization_points(self, optimization_points: usize) -> Self {
        Self {
            optimization_points,
            ..self
        }
    }

    /// Set the maximum cosine distance allowed as a match residual.
    ///
    /// Default: `0.001`
    pub fn cosine_distance_threshold(self, cosine_distance_threshold: f64) -> Self {
        Self {
            cosine_distance_threshold,
            ..self
        }
    }

    /// Adds a new feed with the given intrinsics.
    pub fn insert_feed(&mut self, intrinsics: CameraIntrinsicsK1Distortion) -> usize {
        self.feeds.insert(Feed {
            intrinsics,
            frames: vec![],
        })
    }

    /// Add frame.
    ///
    /// This may perform camera tracking, and thus will take a while.
    ///
    /// Returns a VSlam::frames index.
    pub fn insert_frame(&mut self, feed: usize, image: &DynamicImage) -> usize {
        let next_id = self.frames.insert(Frame {
            feed,
            features: self.kps_descriptors(&self.feeds[feed].intrinsics, image),
            landmarks: Default::default(),
            covisibilities: Default::default(),
        });
        self.feeds[feed].frames.push(next_id);
        let feed_frames = &self.feeds[feed].frames[..];
        if feed_frames.len() >= 2 {
            let a = feed_frames[feed_frames.len() - 2];
            let b = feed_frames[feed_frames.len() - 1];
            self.try_match(Pair::new(a, b));
        }
        next_id
    }

    fn get_pair(&self, Pair(a, b): Pair) -> Option<(&Frame, &Frame)> {
        Some((self.frames.get(a)?, self.frames.get(b)?))
    }

    /// Attempts to match the pair.
    ///
    /// Returns `true` if successful.
    fn try_match(&mut self, pair: Pair) -> bool {
        // First see if the covisibility has already been computed, and if so, don't recompute it.
        if let Some(outcome) = self
            .covisibilities
            .get(&pair)
            .map(|c| c.as_ref().map(|_| pair))
        {
            return outcome.is_some();
        }

        // Get the two frames.
        let (a, b) = self.get_pair(pair).expect("tried to get an invalid pair");

        // Generate the outcome.
        let outcome = self.create_init_covisibility(a, b);
        let success = outcome.is_some();

        // Add the outcome.
        self.add_covisibility_outcome(pair, outcome);

        success
    }

    /// Joins two landmarks together.
    ///
    /// Returns the VSlam::landmarks index of the combined landmark.
    fn join_landmarks(&mut self, lm_aix: usize, lm_bix: usize) -> usize {
        // We will move each observance from b to a.
        let lm_b = self.landmarks.remove(lm_bix);
        for (&frame, &feature) in &lm_b.observances {
            // Point the landmark referenced in the frame to landmark a.
            *self.frames[frame]
                .landmarks
                .get_mut(&feature)
                .expect("observance from landmark b in join_landmarks was invalid") = lm_aix;
            // Insert the observance into landmark a.
            self.landmarks[lm_aix].observances.insert(frame, feature);
        }
        lm_aix
    }

    /// Adds a new landmark from a match and a frame pair.
    ///
    /// Returns the VSlam::landmarks index.
    fn add_landmark(&mut self, pair: Pair, FeatureMatch(aix, bix): FeatureMatch<usize>) -> usize {
        let mut observances = HashMap::new();
        observances.insert(pair.0, aix);
        observances.insert(pair.1, bix);
        let lmix = self.landmarks.insert(Landmark { observances });
        self.frames[pair.0].landmarks.insert(aix, lmix);
        self.frames[pair.1].landmarks.insert(bix, lmix);
        lmix
    }

    /// Adds an observance to the frame and the landmark.
    ///
    /// Returns the VSlam::landmarks index.
    fn add_landmark_observance(&mut self, lmix: usize, frame: usize, feature: usize) -> usize {
        self.frames[frame].landmarks.insert(feature, lmix);
        self.landmarks[lmix].observances.insert(frame, feature);
        lmix
    }

    /// Takes a match and, if it is not already known, adds the observances to the landmark and frame,
    /// combining landmarks as needed.
    ///
    /// Returns the VSlam::landmarks index.
    fn add_match_to_landmarks(
        &mut self,
        pair: Pair,
        FeatureMatch(aix, bix): FeatureMatch<usize>,
    ) -> usize {
        let lm_a = self.frames[pair.0].landmarks.get(&aix).copied();
        let lm_b = self.frames[pair.1].landmarks.get(&bix).copied();

        match (lm_a, lm_b) {
            (Some(lm_aix), Some(lm_bix)) => {
                // If the landmarks are the same, we have found that both frames already share
                // the same landmark, and in this case there is no need to add the observances.
                // However, if the landmarks are different, we must join them together.
                if lm_aix != lm_bix {
                    self.join_landmarks(lm_aix, lm_bix)
                } else {
                    lm_aix
                }
            }
            (Some(lmix), None) => self.add_landmark_observance(lmix, pair.1, bix),
            (None, Some(lmix)) => self.add_landmark_observance(lmix, pair.0, aix),
            (None, None) => self.add_landmark(pair, FeatureMatch(aix, bix)),
        }
    }

    fn add_covisibility_outcome(&mut self, pair: Pair, covisibility: Option<Covisibility>) {
        if let Some(covisibility) = covisibility {
            // First handle updating all the landmarks and frame landmarks.
            for &m in &covisibility.matches {
                self.add_match_to_landmarks(pair, m);
            }
            // Add it to the covisibility map.
            self.covisibilities.insert(pair, Some(covisibility));
            // Add the covisibility to the first frame.
            self.frames[pair.0].covisibilities.insert(pair);
        } else {
            // Put a None to indicate the failure.
            self.covisibilities.insert(pair, None);
        }
    }

    /// This creates a covisibility between frames `a` and `b` using the essential matrix estimator.
    ///
    /// This method resolves to an undefined scale, and thus is only appropriate for initialization.
    fn create_init_covisibility(&self, a: &Frame, b: &Frame) -> Option<Covisibility> {
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |&FeatureMatch(aix, bix)| FeatureMatch(a.keypoint(aix), b.keypoint(bix));

        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let matches: Vec<FeatureMatch<usize>> = symmetric_matching(a, b)
            .filter(|&(_, distance)| distance < self.match_threshold)
            .map(|(m, _)| m)
            .collect();

        info!("estimate essential");

        // Estimate the essential matrix and retrieve the inliers
        let (essential, inliers) = self.consensus.borrow_mut().model_inliers(
            &self.essential_estimator,
            matches
                .iter()
                .map(match_ix_kps)
                .collect::<Vec<_>>()
                .iter()
                .copied(),
        )?;
        // Reconstitute only the inlier matches into a matches vector.
        let matches: Vec<FeatureMatch<usize>> = inliers.into_iter().map(|ix| matches[ix]).collect();

        info!("perform chirality test");

        // Perform a chirality test to retain only the points in front of both cameras.
        let (pose, inliers) = essential
            .pose_solver()
            .solve_unscaled_inliers(self.triangulator.clone(), matches.iter().map(match_ix_kps))?;
        let matches = inliers
            .iter()
            .map(|&inlier| matches[inlier])
            .collect::<Vec<_>>();

        info!("get optimization matches");

        // Select a random sample of 32 points to use for optimization.
        let opti_matches: Vec<FeatureMatch<NormalizedKeyPoint>> = matches
            .choose_multiple(&mut *self.rng.borrow_mut(), self.optimization_points)
            .map(match_ix_kps)
            .collect();

        info!("performing Levenberg-Marquardt");

        let lm = LevenbergMarquardt::new();
        let (tvo, termination) = lm.minimize(TwoViewOptimizer::new(
            opti_matches.iter().copied(),
            pose.0,
            self.triangulator.clone(),
        ));
        let pose = tvo.pose;

        info!(
            "Levenberg-Marquardt terminated with reason {:?}",
            termination
        );

        info!("filtering matches using cosine distance");

        // Filter outlier matches based on cosine distance.
        let matches: Vec<FeatureMatch<usize>> = matches
            .iter()
            .filter_map(|m| {
                let FeatureMatch(a, b) = match_ix_kps(m);
                self.triangulator
                    .triangulate_relative(pose, a, b)
                    .map(|point_a| {
                        let point_b = pose.transform(point_a);
                        1.0 - point_a.coords.normalize().dot(&a.bearing()) + 1.0
                            - point_b.coords.normalize().dot(&b.bearing())
                    })
                    .and_then(|residual| {
                        if residual < self.cosine_distance_threshold {
                            Some(*m)
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Add the new covisibility.
        Some(Covisibility { pose, matches })
    }

    fn kps_descriptors(
        &self,
        intrinsics: &CameraIntrinsicsK1Distortion,
        image: &DynamicImage,
    ) -> Features {
        let (kps, ds) = akaze::Akaze::new(self.akaze_threshold).extract(image);
        kps.into_iter()
            .zip(ds)
            .map(|(kp, d)| (intrinsics.calibrate(kp), d))
            .collect()
    }

    fn covisibility_keypoint_matches<'a>(
        &'a self,
        pair: Pair,
        covisibility: &'a Covisibility,
    ) -> impl Iterator<Item = FeatureMatch<NormalizedKeyPoint>> + 'a {
        covisibility
            .matches
            .iter()
            .map(move |&FeatureMatch(aix, bix)| {
                FeatureMatch(
                    self.frames[pair.0].keypoint(aix),
                    self.frames[pair.1].keypoint(bix),
                )
            })
    }

    pub fn export_covisibility(&self, pair: Pair, path: impl AsRef<Path>) {
        // Get the covisibility object.
        let covisibility = self
            .covisibilities
            .get(&pair)
            .expect("covisibility not found")
            .as_ref()
            .expect("covisibility was found to have been a failure");

        // Output point cloud.
        let points: Vec<Point3<f64>> = self
            .covisibility_keypoint_matches(pair, covisibility)
            .map(|FeatureMatch(a, b)| {
                self.triangulator
                    .triangulate_relative(covisibility.pose, a, b)
                    .unwrap()
                    .0
            })
            .collect();
        crate::export::export(std::fs::File::create(path).unwrap(), points);
    }

    pub fn export_reconstruction_at(
        &self,
        start: usize,
        min_observances: usize,
        path: impl AsRef<Path>,
    ) {
        let (frame_poses, _) = self.frame_graph(start..self.frames.len());

        // Output point cloud.
        let points: Vec<Point3<f64>> = self
            .landmarks
            .iter()
            .filter_map(|(_, lm)| {
                if lm.observances.len() >= min_observances {
                    self.triangulator
                        .triangulate_observances(lm.observances.iter().filter_map(
                            |(&frame, &feature)| {
                                frame_poses.get(&frame).map(|&pose| {
                                    (pose.into(), self.frames[frame].features[feature].0)
                                })
                            },
                        ))
                } else {
                    None
                }
            })
            .map(|WorldPoint(p)| p)
            .collect();
        crate::export::export(std::fs::File::create(path).unwrap(), points);
    }

    /// Turn frame IDs in arbitrary order and which may not be connected into one graph starting with the first frame.
    fn frame_graph(
        &self,
        frames: impl Iterator<Item = usize>,
    ) -> (BTreeMap<usize, WorldPose>, Vec<Pair>) {
        // Find all the frame IDs corresponding to the landmarks and put them into a BTreeSet so they are in order.
        // They must be in order because all correspondences are a directed acyclic graph from lowest frame to largest frame.
        let frames: BTreeSet<usize> = frames.collect();

        // We start by treating the lowest ID frame as the identity pose, and all others as relative.
        let mut frames = frames.into_iter();
        let mut frame_poses: BTreeMap<usize, WorldPose> = BTreeMap::new();
        let mut pairs: Vec<Pair> = Vec::new();
        frame_poses.insert(frames.next().unwrap(), WorldPose::identity());

        for frame in frames {
            if let Some(prev_frame) = frame_poses.keys().copied().find(|&prev_frame| {
                self.covisibilities
                    .get(&Pair::new(prev_frame, frame))
                    .as_ref()
                    .map(|outcome| outcome.is_some())
                    .unwrap_or(false)
            }) {
                // Get the relative pose transforming the previous camera pose to this camera.
                let relative_pose = self.covisibilities[&Pair::new(prev_frame, frame)]
                    .as_ref()
                    .unwrap()
                    .pose
                    .0;
                let prev_pose = frame_poses[&prev_frame].0;
                // Compute the new transformed pose of this camera.
                let pose = WorldPose(prev_pose * relative_pose);
                // Add the pose to the frame poses.
                frame_poses.insert(frame, pose);
                pairs.push(Pair::new(prev_frame, frame));
            }
        }

        (frame_poses, pairs)
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    pub fn bundle_adjust_highest_observances(&mut self, num_landmarks: usize) {
        self.apply_bundle_adjust(self.compute_bundle_adjust_highest_observances(num_landmarks));
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    ///
    /// Returns a series of camera
    fn compute_bundle_adjust_highest_observances(&self, num_landmarks: usize) -> BundleAdjust {
        // At least one landmark exists or the unwraps below will fail.
        if !self.landmarks.is_empty() {
            info!(
                "attempting to extract {} landmarks from a total of {}",
                num_landmarks,
                self.landmarks.len(),
            );

            // First, we want to find the landmarks with the most observances to optimize the reconstruction.
            // Start by putting all the landmark indices into a BTreeMap with the key as their observances and the value the index.
            let mut landmarks_by_observances: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (observance_num, lmix) in self
                .landmarks
                .iter()
                .map(|(ix, lm)| (lm.observances.len(), ix))
            {
                landmarks_by_observances
                    .entry(observance_num)
                    .or_default()
                    .push(lmix);
            }

            info!(
                "found landmarks with (observances, num) of {:?}",
                landmarks_by_observances
                    .iter()
                    .map(|(ob, v)| (ob, v.len()))
                    .collect::<Vec<_>>()
            );

            // Now the BTreeMap is sorted from smallest number of observances to largest, so take the last indices.
            let opti_landmarks: Vec<usize> = landmarks_by_observances
                .values()
                .rev()
                .flat_map(|v| v.iter().copied())
                .take(num_landmarks)
                .collect();

            // Find all the frame IDs corresponding to the landmarks and convert them to a graph.
            let (frame_poses, pairs) =
                self.frame_graph(opti_landmarks.iter().copied().flat_map(|lm| {
                    self.landmarks[lm]
                        .observances
                        .iter()
                        .map(|(&frame, _)| frame)
                }));

            info!(
                "performing Levenberg-Marquardt on best {} landmarks and {} frames",
                opti_landmarks.len(),
                frame_poses.len(),
            );

            let levenberg_marquardt = LevenbergMarquardt::new();
            let (mvo, termination) = levenberg_marquardt.minimize(ManyViewOptimizer::new(
                frame_poses.values().copied().collect::<Vec<WorldPose>>(),
                opti_landmarks.iter().copied().map(|lm| {
                    frame_poses.keys().map(move |&frame| {
                        self.landmarks[lm]
                            .observances
                            .get(&frame)
                            .map(move |&feature| self.frames[frame].features[feature].0)
                    })
                }),
                self.triangulator.clone(),
            ));
            let poses = mvo.poses;

            info!(
                "Levenberg-Marquardt terminated with reason {:?}",
                termination
            );

            BundleAdjust {
                poses: frame_poses.keys().copied().zip(poses).collect(),
                pairs,
            }
        } else {
            BundleAdjust {
                poses: HashMap::new(),
                pairs: vec![],
            }
        }
    }

    fn apply_bundle_adjust(&mut self, bundle_adjust: BundleAdjust) {
        let BundleAdjust { poses, pairs } = bundle_adjust;
        // Only run if there is at least one pair.
        if !pairs.is_empty() {
            assert_eq!(poses.len() - 1, pairs.len());
            for pair in pairs {
                self.covisibilities
                    .get_mut(&pair)
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .pose = RelativeCameraPose(poses[&pair.0].0.inverse() * poses[&pair.1].0);
            }
        }
    }
}

fn matching(
    a_descriptors: impl Iterator<Item = BitArray<64>>,
    b_descriptors: impl Iterator<Item = BitArray<64>> + Clone,
) -> Vec<(usize, usize)> {
    a_descriptors
        .map(|a| {
            b_descriptors
                .clone()
                .map(|b| a.distance(&b))
                .enumerate()
                .min_by_key(|&(_, d)| d)
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn symmetric_matching<'a>(
    a: &'a Frame,
    b: &'a Frame,
) -> impl Iterator<Item = (FeatureMatch<usize>, usize)> + 'a {
    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a.descriptors(), b.descriptors());
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b.descriptors(), a.descriptors());
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, (bix, distance))| {
            // Does the feature in b match with this feature too?
            let is_symmetric = reverse_matches[bix].0 == aix;
            if is_symmetric {
                Some((FeatureMatch(aix, bix), distance))
            } else {
                None
            }
        })
}
