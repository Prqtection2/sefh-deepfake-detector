import cv2
import numpy as np
import os
import glob
import pandas as pd
import joblib
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_FOLDER = "data"
REAL_SUBFOLDER = "real"
FAKE_SUBFOLDER = "deepfakes"
CSV_FILE = "svd_magnum_features.csv"
MODEL_FILE = "deepfake_magnum_model.pkl"
MAX_VIDEOS = 10000 
FRAMES_PER_BLOCK = 30 

def extract_magnum_features(video_path):
    try:
        label = 0 if "real" in video_path.lower() else 1
        cap = cv2.VideoCapture(video_path)
        
        frames_gray = []
        frames_chroma = [] 
        edge_energies = [] # New: Gradient Analysis
        patch_glitches = [] # Restored: Spatial Glitch
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count >= FRAMES_PER_BLOCK: break
            
            # 1. GRAYSCALE (For Entropy/Noise)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64)) 
            frames_gray.append(small.flatten())
            
            # 2. CHROMA SVD (The "Queen" Feature)
            if count % 5 == 0:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                _, cr, cb = cv2.split(cv2.resize(ycrcb, (64, 64)))
                frames_chroma.append(cr.flatten()) 

            # 3. SPATIAL GLITCH (Restored from 91% model)
            if count % 5 == 0:
                center = gray[16:48, 16:48]
                s_patch = np.linalg.svd(center, compute_uv=False)
                patch_glitches.append(s_patch[0] / (np.sum(s_patch) + 1e-9))
                
            # 4. GRADIENT SVD (New! Edge Physics)
            # We calculate the edges (Sobel) and run SVD on the Edge Map
            if count % 3 == 0:
                sobelx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # SVD on the Edges
                s_edge = np.linalg.svd(magnitude, compute_uv=False)
                # Edge Complexity: Real edges have energy spread out (Low Ratio)
                # Fake edges are smooth (High Ratio)
                edge_energies.append(s_edge[0] / (np.sum(s_edge) + 1e-9))

            count += 1
        cap.release()
        
        if len(frames_gray) < FRAMES_PER_BLOCK: return None
        
        # --- FEATURE 1: SVD ENTROPY ---
        M = np.array(frames_gray).T 
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        p = S / np.sum(S)
        f1_entropy = -np.sum(p * np.log(p + 1e-12))
        
        # --- FEATURE 2: TAIL ENERGY (The "King" Feature) ---
        f2_tail = np.sum(S[-10:]) / np.sum(S)
        
        # --- FEATURE 3: CHROMA SCORE ---
        if frames_chroma:
            M_c = np.array(frames_chroma).T
            s_c = np.linalg.svd(M_c, compute_uv=False)
            f3_chroma = s_c[0] / (np.sum(s_c) + 1e-9)
        else:
            f3_chroma = 0
            
        # --- FEATURE 4: MAX GLITCH (Restored) ---
        f4_glitch = np.percentile(patch_glitches, 95) if patch_glitches else 0
        
        # --- FEATURE 5: EDGE RANK (New) ---
        f5_edge = np.median(edge_energies) if edge_energies else 0

        return [os.path.basename(video_path), label, f1_entropy, f2_tail, f3_chroma, f4_glitch, f5_edge]

    except Exception as e:
        return None

def main():
    print("Initializing MAGNUM OPUS MODEL (Ensemble + Gradients)...")
    
    # 1. GATHER FILES
    real_vids = glob.glob(os.path.join(VIDEO_FOLDER, REAL_SUBFOLDER, "*.mp4"))
    fake_vids = glob.glob(os.path.join(VIDEO_FOLDER, FAKE_SUBFOLDER, "*.mp4"))
    
    limit = min(len(real_vids), len(fake_vids), MAX_VIDEOS // 2)
    real_vids = real_vids[:limit]
    fake_vids = fake_vids[:limit]
    all_vids = real_vids + fake_vids
    
    print(f"Processing {len(all_vids)} videos...")
    
    # 2. EXTRACT
    processed_files = set()
    if os.path.exists(CSV_FILE):
        try:
            df_exist = pd.read_csv(CSV_FILE)
            processed_files = set(df_exist['filename'].values)
            print(f"Resuming... {len(processed_files)} processed.")
        except: pass
            
    videos_to_do = [v for v in all_vids if os.path.basename(v) not in processed_files]
    
    if videos_to_do:
        print(f"Extracting features for {len(videos_to_do)} new videos...")
        with open(CSV_FILE, 'a') as f:
            if os.path.getsize(CSV_FILE) == 0:
                f.write("filename,label,svd_entropy,tail_energy,chroma_score,spatial_glitch,edge_rank\n")
            
            with ProcessPoolExecutor() as executor:
                results = executor.map(extract_magnum_features, videos_to_do)
                count = 0
                for res in results:
                    if res:
                        line = f"{res[0]},{res[1]},{res[2]:.6f},{res[3]:.6f},{res[4]:.6f},{res[5]:.6f},{res[6]:.6f}\n"
                        f.write(line)
                        f.flush()
                    count += 1
                    if count % 20 == 0: print(f"Progress: {count}/{len(videos_to_do)}", end='\r')

    # 3. ENSEMBLE TRAINING
    print("\nTraining Ensemble Model...")
    df = pd.read_csv(CSV_FILE)
    target_names = set([os.path.basename(v) for v in all_vids])
    df = df[df['filename'].isin(target_names)]
    
    X = df[['svd_entropy', 'tail_energy', 'chroma_score', 'spatial_glitch', 'edge_rank']].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # --- THE VOTING ENSEMBLE ---
    # We combine 3 different brains:
    # 1. Random Forest (Great at rules)
    # 2. Gradient Boosting (Great at patterns)
    # 3. Logistic Regression (Great at simple boundaries)
    
    clf1 = RandomForestClassifier(n_estimators=300, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    clf3 = LogisticRegression(random_state=42, max_iter=1000)
    
    eclf = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)], voting='soft')
    eclf.fit(X_train, y_train)
    
    # 4. RESULTS
    print("\n" + "="*40)
    print(f"MAGNUM OPUS ACCURACY: {accuracy_score(y_test, eclf.predict(X_test))*100:.2f}%")
    print("="*40)
    print(confusion_matrix(y_test, eclf.predict(X_test)))
    
    # Feature Importance (using the GB model as proxy)
    clf2.fit(X_train, y_train)
    print("\nFeature Importance (Proxy):")
    feats = ['SVD Entropy', 'Tail Energy', 'Chroma Score', 'Spatial Glitch', 'Edge Rank']
    for name, imp in zip(feats, clf2.feature_importances_):
        print(f"{name:<20}: {imp:.4f}")
        
    joblib.dump(eclf, MODEL_FILE)

if __name__ == "__main__":
    main()