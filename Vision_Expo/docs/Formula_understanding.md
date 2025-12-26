# Formula Understanding - Technical Deep Dive

This document explains the mathematical formulas and algorithms used in the Vision_Expo volleyball tracking system.

## Table of Contents
1. [Geometric Calculations](#geometric-calculations)
2. [Kalman Filtering](#kalman-filtering)
3. [Hungarian Algorithm & Cost Matrix](#hungarian-algorithm--cost-matrix)
4. [Camera Motion Compensation](#camera-motion-compensation)
5. [Referee Classification](#referee-classification)
6. [Team Assignment](#team-assignment)
7. [Optical Flow](#optical-flow)
8. [Non-Maximum Suppression](#non-maximum-suppression)

---

## 1. Geometric Calculations

### 1.1 Centroid Calculation

The centroid (center point) of a bounding box:

```
centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
```

Where:
- `(x1, y1)` = top-left corner
- `(x2, y2)` = bottom-right corner

**Usage**: Track position representation, distance calculations

---

### 1.2 Intersection over Union (IoU)

Measures overlap between two bounding boxes:

```
IoU = Area(Intersection) / Area(Union)

Intersection:
  x_left = max(x1_a, x1_b)
  y_top = max(y1_a, y1_b)
  x_right = min(x2_a, x2_b)
  y_bottom = min(y2_a, y2_b)

  inter_area = max(0, x_right - x_left) × max(0, y_bottom - y_top)

Union:
  area_a = (x2_a - x1_a) × (y2_a - y1_a)
  area_b = (x2_b - x1_b) × (y2_b - y1_b)
  union_area = area_a + area_b - inter_area

IoU = inter_area / (union_area + ε)
```

Where `ε = 1e-8` prevents division by zero.

**Range**: [0, 1], where 1 = perfect overlap

**Usage**: Track-detection matching, NMS

---

### 1.3 Euclidean Distance

Distance between two points:

```
distance = √[(x2 - x1)² + (y2 - y1)²]
```

Or using numpy:
```python
distance = np.linalg.norm(point_a - point_b)
```

**Usage**: Position matching, referee zone checks

---

## 2. Kalman Filtering

### 2.1 State Space Model

6-dimensional state vector:
```
x = [px, py, vx, vy, ax, ay]ᵀ
```

Where:
- `px, py` = position (x, y coordinates)
- `vx, vy` = velocity
- `ax, ay` = acceleration

---

### 2.2 State Transition Matrix (F)

Constant acceleration model with time step `dt = 1`:

```
F = ┌                           ┐
    │ 1   0   dt  0   0.5dt²  0  │
    │ 0   1   0   dt  0   0.5dt² │
    │ 0   0   1   0   dt   0     │
    │ 0   0   0   1   0    dt    │
    │ 0   0   0   0   1    0     │
    │ 0   0   0   0   0    1     │
    └                           ┘
```

**Physics derivation**:
- `position_new = position + velocity×dt + 0.5×acceleration×dt²`
- `velocity_new = velocity + acceleration×dt`
- `acceleration_new = acceleration` (constant)

---

### 2.3 Measurement Matrix (H)

Only observe position (not velocity/acceleration):

```
H = ┌                   ┐
    │ 1  0  0  0  0  0  │
    │ 0  1  0  0  0  0  │
    └                   ┘
```

Maps 6D state to 2D observation: `z = H × x`

---

### 2.4 Prediction Step

```
x̂ₖ₊₁|ₖ = F × x̂ₖ|ₖ               (predicted state)
Pₖ₊₁|ₖ = F × Pₖ|ₖ × Fᵀ + Q      (predicted covariance)
```

Where:
- `P` = error covariance matrix
- `Q` = process noise covariance

---

### 2.5 Update Step

```
Innovation:
  yₖ = zₖ - H × x̂ₖ|ₖ₋₁

Innovation covariance:
  Sₖ = H × Pₖ|ₖ₋₁ × Hᵀ + R

Kalman gain:
  Kₖ = Pₖ|ₖ₋₁ × Hᵀ × Sₖ⁻¹

Updated state:
  x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ × yₖ

Updated covariance:
  Pₖ|ₖ = (I - Kₖ × H) × Pₖ|ₖ₋₁
```

Where `R` = measurement noise covariance

---

### 2.6 Tuning Parameters

**For players/referees**:
```python
kf.P *= 10      # Initial uncertainty
kf.R *= 2       # Measurement noise
kf.Q *= 0.05    # Process noise (low - smooth motion)
```

**For ball** (more erratic):
```python
kf.P *= 50      # Higher initial uncertainty
kf.R *= 0.5     # Trust measurements more
kf.Q *= 2.0     # Higher process noise
```

---

## 3. Hungarian Algorithm & Cost Matrix

### 3.1 Cost Matrix Construction

For each track-detection pair:

```
Cost(i,j) = 1 - CombinedScore(i,j)

CombinedScore = w₁×IoU + w₂×DistScore + w₃×SizeScore

Where:
  w₁ = 0.5  (IoU weight)
  w₂ = 0.3  (distance weight)
  w₃ = 0.2  (size weight)
```

---

### 3.2 Distance Score

```
DistScore = max(0, 1 - distance / max_distance)

distance = ||centroid_predicted - centroid_detected||
max_distance = 200 pixels (scaled)
```

**Range**: [0, 1], where 1 = zero distance

---

### 3.3 Size Similarity Score

```
SizeScore = min(area_pred, area_det) / max(area_pred, area_det)
```

**Range**: [0, 1], where 1 = identical size

---

### 3.4 Hungarian Algorithm

Given cost matrix `C[n_tracks × n_detections]`, find assignment that minimizes total cost:

```
minimize: Σᵢ Σⱼ (xᵢⱼ × Cᵢⱼ)

subject to:
  Σⱼ xᵢⱼ ≤ 1  for all i  (each track assigned ≤ 1 detection)
  Σᵢ xᵢⱼ ≤ 1  for all j  (each detection assigned ≤ 1 track)
  xᵢⱼ ∈ {0,1}
```

**Implementation**: `scipy.optimize.linear_sum_assignment(cost_matrix)`

**Threshold**: Only accept matches with `cost < 0.65` (score > 0.35)

---

## 4. Camera Motion Compensation

### 4.1 Homography Estimation

Homography `H` transforms points from frame t to frame t+1:

```
┌   ┐       ┌           ┐ ┌   ┐
│ x'│       │ h₁₁ h₁₂ h₁₃│ │ x │
│ y'│ = λ × │ h₂₁ h₂₂ h₂₃│ │ y │
│ 1 │       │ h₃₁ h₃₂ h₃₃│ │ 1 │
└   ┘       └           ┘ └   ┘

After normalization:
  x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
  y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

---

### 4.2 Feature Matching

**SIFT/ORB Feature Detection**:
1. Detect keypoints in frames t and t+1
2. Compute descriptors for each keypoint
3. Match descriptors using k-NN (k=2)

**Lowe's Ratio Test**:
```
if distance(match₁) < 0.7 × distance(match₂):
    accept match₁ as good match
```

Filters ambiguous matches.

---

### 4.3 RANSAC Homography

Robust estimation despite outliers:

```
Repeat N iterations:
  1. Randomly sample 4 point pairs
  2. Compute homography H from sample
  3. Count inliers: points where reprojection_error < threshold
  4. Keep H with most inliers

threshold = 5.0 pixels
```

**Confidence**:
```
confidence = num_inliers / num_total_matches
```

---

## 5. Referee Classification

### 5.1 Position-Based Scoring

For each candidate track at referee zone:

```
Score = DistanceScore + StabilityBonus + HistoryBonus + AgeBonus

DistanceScore:
  score += (1 - min(distance/radius, 1)) × 10.0

StabilityBonus:
  if track.is_stable():
      score += 3.0

HistoryBonus:
  avg_position = mean(track.history[-15:])
  avg_distance = ||avg_position - zone_center||
  if avg_distance < radius:
      score += 5.0

AgeBonus:
  if track.age > 30:
      score += 2.0
```

**Maximum score**: ~20 points
**Threshold**: 8.0 points required for assignment

---

### 5.2 Position Check

```
is_in_zone = ||track.centroid - zone_position|| < zone_radius

zone_radius = 260 pixels (scaled)
```

---

### 5.3 Hysteresis Mechanism

Prevents oscillation between referee/player:

```
if not is_in_zone:
    strike_count += 1
else:
    strike_count = 0

if strike_count >= 10:
    demote_to_player()
```

Requires 10 consecutive frames outside zone before demotion.

---

### 5.4 Camera Motion Update

Adjust zone position for camera movement:

```
┌    ┐       ┌   ┐
│ x' │       │ x │
│ y' │ = H × │ y │
│ w' │       │ 1 │
└    ┘       └   ┘

Normalize:
  x_new = x' / w'
  y_new = y' / w'

Validate:
  if 0 ≤ x_new < width and 0 ≤ y_new < height:
      zone_position = [x_new, y_new]
```

---

## 6. Team Assignment

### 6.1 Cross Product Method

Determine which side of net line a player is on:

```
Net vector:
  v_net = [net_p2.x - net_p1.x, net_p2.y - net_p1.y]

Player vector (from net_p1):
  v_player = [player.x - net_p1.x, player.y - net_p1.y]

Cross product (2D):
  cross = v_net.x × v_player.y - v_net.y × v_player.x

Team assignment:
  if cross > 0: Team A
  else: Team B
```

**Intuition**: Positive cross product = left side, negative = right side

---

### 6.2 Voting System

Stabilize team assignment over time:

```
team_votes = deque(maxlen=15)  # Rolling window

For each frame:
  team_votes.append(new_team)

If len(team_votes) >= 5:
  final_team = most_common(team_votes)
```

Prevents flickering due to temporary net tracking errors.

---

## 7. Optical Flow

### 7.1 Lucas-Kanade Method

Track points between frames using pyramidal LK:

```
Parameters:
  winSize = (21, 21)           # Search window
  maxLevel = 3                 # Pyramid levels
  criteria = (EPS|COUNT, 30, 0.01)

For each point p in frame_t:
  Find p' in frame_t+1 that minimizes:
    Σ [I_t(x) - I_t+1(x + d)]²

  over window W centered at p
```

Where:
- `I_t` = intensity in frame t
- `d` = displacement vector

---

### 7.2 Stability Check

```
Motion per point:
  motion[i] = ||point_new[i] - point_old[i]||

Average motion:
  avg_motion = mean(motion)

Sudden jump detection:
  if avg_motion > threshold:
      mark_unstable()

threshold = 50 pixels (scaled)
```

---

### 7.3 Confidence Estimation

```
confidence = max(0.1, 1.0 - avg_motion / 30)
```

**Range**: [0.1, 1.0]
- High confidence (near 1.0) = small motion = stable tracking
- Low confidence (near 0.1) = large motion = potential failure

---

## 8. Non-Maximum Suppression

### 8.1 Greedy NMS Algorithm

Remove overlapping detections:

```
Input: boxes = [(x1, y1, x2, y2, score), ...]
       iou_threshold = 0.4

1. Sort boxes by score (descending)
2. While boxes remain:
     a. Select highest-scoring box
     b. Add to keep list
     c. Compute IoU with all remaining boxes
     d. Remove boxes with IoU > threshold
3. Return kept boxes
```

---

### 8.2 IoU Calculation in NMS

```
For box i and all boxes j in remaining:
  inter_area = (min(x2_i, x2_j) - max(x1_i, x1_j)) ×
               (min(y2_i, y2_j) - max(y1_i, y1_j))

  union_area = area_i + area_j - inter_area

  IoU = inter_area / union_area

  if IoU > 0.4:
      remove box j
```

---

## 9. Stability Metrics

### 9.1 Position Variance

Measures track movement consistency:

```
recent_positions = history[-10:]  # Last 10 positions

position_variance = mean(var(positions, axis=0))

Stable if:
  position_variance < 50
```

Low variance = consistent position = stable track

---

### 9.2 Size Variance

Measures bounding box size consistency:

```
recent_areas = [box_area(bbox) for bbox in history[-5:]]

size_variance = var(recent_areas)
```

Used to detect tracking instability.

---

### 9.3 Track Confirmation

```
Track is confirmed if:
  (hits >= 3) AND (age >= 3)

hits = number of successful matches
age = frames since creation
```

Prevents false positives from spurious detections.

---

## 10. Scale Adaptation

### 10.1 Resolution Scaling

Adapt all distances/sizes to video resolution:

```
scale_x = actual_width / reference_width
scale_y = actual_height / reference_height
scale_avg = (scale_x + scale_y) / 2

For any reference distance D:
  scaled_distance = D × scale_avg

For any reference point (x, y):
  scaled_point = (x × scale_x, y × scale_y)
```

**Reference**: 1920×1080 (Full HD)

---

### 10.2 Font and Line Scaling

```
scaled_font_size = max(0.3, base_size × scale_avg)
scaled_thickness = max(1, int(base_thickness × scale_avg))
```

Ensures readable text and visible lines at any resolution.

---

## Summary of Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| IoU threshold (players) | 0.30 | Track-detection matching |
| IoU threshold (refs) | 0.25 | More lenient for referees |
| IoU threshold (ball) | 0.15 | Ball moves fast |
| Max age (players) | 30 frames | Remove lost tracks |
| Max age (refs) | 120 frames | Refs more persistent |
| Referee zone radius | 260 px | Position lock area |
| Strike threshold | 10 frames | Demotion hysteresis |
| Confirmation threshold | 3 hits in 3 frames | Track validation |
| Cost threshold | 0.65 | Assignment acceptance |
| Optical flow window | 21×21 | LK search area |
| Homography threshold | 5.0 px | RANSAC inlier threshold |

---

## References

1. **Kalman Filter**: Welch & Bishop, "An Introduction to the Kalman Filter" (2006)
2. **Hungarian Algorithm**: Kuhn, "The Hungarian Method for the Assignment Problem" (1955)
3. **Homography**: Hartley & Zisserman, "Multiple View Geometry in Computer Vision" (2003)
4. **Lucas-Kanade**: Lucas & Kanade, "An Iterative Image Registration Technique" (1981)
5. **NMS**: Neubeck & Van Gool, "Efficient Non-Maximum Suppression" (2006)

---

**Document Version**: 1.0.0
**Last Updated**: 2024
