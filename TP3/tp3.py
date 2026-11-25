# ============================================================
# Part 0: Load Two Images
# ============================================================

import cv2
import numpy as np

reference = cv2.imread("ref1.jpg")
scene = cv2.imread("scene1.jpg")
if reference is None:
    raise FileNotFoundError("reference.jpg not found")

if scene is None :
    raise FileNotFoundError("scene.jpg not found")

ref_gray =cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

ref_float =np.float32(ref_gray)



# ============================================================
# Part 1: Harris Corners
# ============================================================
harris =cv2.cornerHarris(ref_float,2,3,0.04)
ref_harris = reference.copy()
threshold = 0.01 *harris.max()

ref_harris[harris > threshold] = [0, 0, 255]

cv2.imshow("ref", reference)
cv2.imshow("ref_harris", ref_harris)


# ============================================================
# Part 2: SIFT Detection
# ============================================================

sift = cv2.SIFT_create()

kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)

print("Reference keypoints:", len(kp_ref))
print("Scene keypoints:", len(kp_scene))
print("Descriptor dimensions:", des_ref.shape)

ref_features = cv2.drawKeypoints(
    reference,
    kp_ref,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

scene_features = cv2.drawKeypoints(
    scene,
    kp_scene,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("ref", ref_features)
cv2.imshow("ref_haris", scene_features)

# ============================================================
# Part 3: ORB Detection
# ============================================================

orb = cv2.ORB_create(nfeatures=1000)

orb_kp_ref, orb_des_ref = orb.detectAndCompute(ref_gray, None)

orb_kp_scene, orb_des_scene = orb.detectAndCompute(scene_gray, None)

print("ORB reference:", len(orb_kp_ref))
print("ORB scene:", len(orb_kp_scene))

ref_features = cv2.drawKeypoints(
    reference,
    orb_kp_ref,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

scene_features = cv2.drawKeypoints(
    scene,
    orb_kp_scene,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("ref", ref_features)
cv2.imshow("ref_haris", scene_features)


# ============================================================
# Part 4: SIFT Brute-Force Matching
# ============================================================
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_ref,des_scene)
matches = sorted(matches, key=lambda x:x.distance)

matches =cv2.drawMatches(
    reference,kp_ref,scene,kp_scene,matches[:50],None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imshow("SIFT_matched", matches)

# ============================================================
# Part 5: Ratio Test
# ============================================================
bf = cv2.BFMatcher()

knn_matches = bf.knnMatch(des_ref, des_scene, k=2)

good_matches = []

for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        
print("Good matches:",len(good_matches))

good_image = cv2.drawMatches(
    reference,kp_ref,scene,kp_scene,good_matches,None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    
)
cv2.imshow("SIFT_matched", good_image)


cv2.waitKey(0)
cv2.destroyAllWindows()