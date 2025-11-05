"""
Fixed Ilastik Data Loader for YOLO Training
Maps training images to their corresponding ILASTIK segmentation labels
"""

from pathlib import Path
from typing import List, Tuple
import os

class IlastikDataMapper:
    """Maps training images to their corresponding Ilastik segmentation labels"""
    
    def __init__(self):
        # Base paths for Google Colab
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.script_dir)))
        self.data_dir = os.path.join(self.project_root,'CODE','DL MODELS (copy)')
        self.dataset_path = os.path.join(self.data_dir,"Dataset")
        self.labels_path = os.path.join(self.data_dir,"Labels","Labels")
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Labels path: {self.labels_path}")
    
    def load_data_pairs(self, imaging_modality: str) -> List[Tuple[str, str]]:
        """
        Load all image-label pairs for a specific imaging modality
        
        Args:
            imaging_modality: 'AFM', 'CRYO-SEM', 'STED', or 'CONFOCAL'
            
        Returns:
            List of (image_path, label_path) tuples
        """
        pairs = []
        
        if imaging_modality.upper() == 'AFM':
            pairs = self._load_afm_pairs()
        elif imaging_modality.upper() == 'CRYO-SEM':
            pairs = self._load_cryosem_pairs()
        elif imaging_modality.upper() == 'STED':
            pairs = self._load_sted_pairs()
        elif imaging_modality.upper() == 'CONFOCAL':
            pairs = self._load_confocal_pairs()
        else:
            print(f"Unknown modality: {imaging_modality}")
            return []
        
        print(f"Found {len(pairs)} pairs for {imaging_modality}")
        return pairs
    
    def _load_afm_pairs(self) -> List[Tuple[str, str]]:
        """Load AFM image-label pairs"""
        pairs = []
        
        # AFM training image directories
        training_folders = [
            self.dataset_path / "AFM folder" / "AFM Training" / "1%",
            self.dataset_path / "AFM folder" / "AFM Training" / "1.5%",
            self.dataset_path / "AFM folder" / "AFM Training" / "2%"
        ]
        
        # AFM label directories
        label_folders = {
            "1%": self.labels_path / "ILASTIK [1%] AFM" / "ILASTIK [1%] AFM_aug_inverted",
            "1.5%": self.labels_path / "ILASTIK [1.5%] AFM" / "ILASTIK [1.5%] AFM_aug_inverted",
            "2%": self.labels_path / "ILASTIK [2%] AFM" / "ILASTIK [2%] AFM_aug_inverted"
        }
        
        for training_folder in training_folders:
            if not training_folder.exists():
                print(f"Warning: Training folder not found: {training_folder}")
                continue
            
            # Get the concentration folder name (1%, 1.5%, 2%)
            concentration = training_folder.name
            label_folder = label_folders.get(concentration)
            
            if not label_folder or not label_folder.exists():
                print(f"Warning: Label folder not found for {concentration}")
                continue
            
            # Get all training images
            image_files = list(training_folder.glob("*.tif")) + list(training_folder.glob("*.tiff"))
            
            for img_path in image_files:
                # Match image to label
                # Image: pp_1percent_hflip.tif
                # Label: pp_1percent_ILASTIK_hflip.tif
                base_name = img_path.stem  # e.g., pp_1percent_hflip
                
                # Try to find corresponding label
                label_name = base_name.replace("_hflip", "_ILASTIK_hflip") \
                                      .replace("_vflip", "_ILASTIK_vflip") \
                                      .replace("_rot90", "_ILASTIK_rot90") \
                                      .replace("_rot180", "_ILASTIK_rot180") \
                                      .replace("_rot270", "_ILASTIK_rot270")
                
                # If no augmentation suffix, just add _ILASTIK
                if "_ILASTIK" not in label_name:
                    label_name = base_name + "_ILASTIK"
                
                label_path = label_folder / f"{label_name}.tif"
                
                if label_path.exists():
                    pairs.append((str(img_path), str(label_path)))
                else:
                    # Try without ILASTIK in the middle
                    label_path_alt = label_folder / f"{base_name}.tif"
                    if label_path_alt.exists():
                        pairs.append((str(img_path), str(label_path_alt)))
                    else:
                        print(f"Warning: No label found for {img_path.name}")
        
        return pairs
    
    def _load_cryosem_pairs(self) -> List[Tuple[str, str]]:
        """Load CRYO-SEM image-label pairs"""
        pairs = []
        
        # CRYO-SEM training directories
        magnifications = ["x1000", "x3000", "x10000", "x30000", "x60000"]
        
        for mag in magnifications:
            training_folder = self.dataset_path / "Cryo-sem Folder" / "CRYO-SEM_Training" / mag
            
            if not training_folder.exists():
                print(f"Warning: Training folder not found: {training_folder}")
                continue
            
            # Find corresponding label folder
            label_folder = None
            if mag == "x1000":
                label_folder = self.labels_path / "Ilastik [x1000] cryo-sem" / "Ilastik [x1000] cryo-sem_aug_inverted"
            elif mag == "x3000":
                label_folder = self.labels_path / "3000x [ILASTIK] AFM" / "3000x [ILASTIK] AFM_aug_inverted"
            elif mag == "x10000":
                label_folder = self.labels_path / "ILASTIK [10000] cryo-sem" / "ILASTIK [10000] cryo-sem_aug_inverted"
            elif mag == "x30000":
                label_folder = self.labels_path / "ILASTIK [x30000] cryo-sem" / "ILASTIK [x30000] cryo-sem_aug_inverted"
            elif mag == "x60000":
                label_folder = self.labels_path / "ILASTIK [60000] cryo-sem" / "ILASTIK [60000] cryo-sem_aug_inverted"
            
            if not label_folder or not label_folder.exists():
                print(f"Warning: Label folder not found for {mag}")
                continue
            
            # Get all training images
            image_files = list(training_folder.glob("*.tif")) + list(training_folder.glob("*.tiff"))
            
            for img_path in image_files:
                # Match image to label
                # Image: CRYO-SEM pp x1000c_hflip.tif
                # Label: CRYO-SEM pp x1000c_ILASTIK_hflip.tif
                base_name = img_path.stem
                
                # Insert _ILASTIK before augmentation suffix
                label_name = base_name.replace("_hflip", "_ILASTIK_hflip") \
                                      .replace("_vflip", "_ILASTIK_vflip") \
                                      .replace("_rot90", "_ILASTIK_rot90") \
                                      .replace("_rot180", "_ILASTIK_rot180") \
                                      .replace("_rot270", "_ILASTIK_rot270")
                
                label_path = label_folder / f"{label_name}.tif"
                
                if label_path.exists():
                    pairs.append((str(img_path), str(label_path)))
                else:
                    print(f"Warning: No label found for {img_path.name}")
        
        return pairs
    
    def _load_sted_pairs(self) -> List[Tuple[str, str]]:
        """Load STED image-label pairs"""
        pairs = []
        
        concentrations = ["0.375%", "1%"]
        
        for conc in concentrations:
            training_folder = self.dataset_path / "STED Folder" / "STED Training" / conc
            
            if not training_folder.exists():
                print(f"Warning: Training folder not found: {training_folder}")
                continue
            
            # Find label folder
            if conc == "0.375%":
                label_folder = self.labels_path / "STED" / "0.375 STED" / "ILASTIK" / "ILASTIK_aug_inverted"
            else:
                label_folder = self.labels_path / "STED" / "1 STED" / "ILASTIK" / "ILASTIK_aug_inverted"
            
            if not label_folder.exists():
                print(f"Warning: Label folder not found for STED {conc}")
                continue
            
            # Get all training images
            image_files = list(training_folder.glob("*.tif")) + list(training_folder.glob("*.tiff"))
            
            for img_path in image_files:
                # Match image to label
                # Image: [PP]0_375perc_STED_2-iter10_hflip.tif
                # Label: 0_375perc_STED_2-iter20_ILASTIK_hflip.tif
                base_name = img_path.stem
                
                # Remove [PP] prefix and change iter10 to iter20
                label_name = base_name.replace("[PP]", "").replace("iter10", "iter20")
                
                # Insert _ILASTIK before augmentation
                label_name = label_name.replace("_hflip", "_ILASTIK_hflip") \
                                       .replace("_vflip", "_ILASTIK_vflip") \
                                       .replace("_rot90", "_ILASTIK_rot90") \
                                       .replace("_rot180", "_ILASTIK_rot180") \
                                       .replace("_rot270", "_ILASTIK_rot270")
                
                label_path = label_folder / f"{label_name}.tif"
                
                if label_path.exists():
                    pairs.append((str(img_path), str(label_path)))
                else:
                    print(f"Warning: No label found for {img_path.name}")
        
        return pairs
    
    def _load_confocal_pairs(self) -> List[Tuple[str, str]]:
        """Load CONFOCAL image-label pairs"""
        pairs = []
        
        concentrations = ["0.375%", "1%"]
        
        for conc in concentrations:
            training_folder = self.dataset_path / "Confocal folder" / "CONFOCAL Training" / conc
            
            if not training_folder.exists():
                print(f"Warning: Training folder not found: {training_folder}")
                continue
            
            # Find label folder
            if conc == "0.375%":
                label_folder = self.labels_path / "Conf" / "0.375 conf" / "ILASTIK" / "ILASTIK_aug_inverted"
            else:
                label_folder = self.labels_path / "Conf" / "1 conf" / "ILASTIK" / "ILASTIK_aug_inverted"
            
            if not label_folder.exists():
                print(f"Warning: Label folder not found for CONFOCAL {conc}")
                continue
            
            # Get all training images
            image_files = list(training_folder.glob("*.tif")) + list(training_folder.glob("*.tiff"))
            
            for img_path in image_files:
                # Match image to label
                # Image: [PP]0_375perc_Confocal_2-iter6_hflip.tif
                # Label: pp 0_375perc_Confocal_2-iter20_ILASTIK_hflip.tif OR 0_375perc_Confocal_3-iter20_ILASTIK_hflip.tif
                base_name = img_path.stem
                
                # Remove [PP] and change iter6 to iter20
                label_name = base_name.replace("[PP]", "").replace("iter6", "iter20")
                
                # Insert _ILASTIK before augmentation
                label_name = label_name.replace("_hflip", "_ILASTIK_hflip") \
                                       .replace("_vflip", "_ILASTIK_vflip") \
                                       .replace("_rot90", "_ILASTIK_rot90") \
                                       .replace("_rot180", "_ILASTIK_rot180") \
                                       .replace("_rot270", "_ILASTIK_rot270")
                
                # Try with and without "pp " prefix
                label_path1 = label_folder / f"{label_name}.tif"
                label_path2 = label_folder / f"pp {label_name}.tif"
                
                if label_path1.exists():
                    pairs.append((str(img_path), str(label_path1)))
                elif label_path2.exists():
                    pairs.append((str(img_path), str(label_path2)))
                else:
                    print(f"Warning: No label found for {img_path.name}")
        
        return pairs
    
    def map_training_image_to_label(self, image_path: str, imaging_modality: str) -> str:
        """
        Map a single training image to its corresponding label
        (Kept for backward compatibility)
        """
        pairs = self.load_data_pairs(imaging_modality)
        for img, lbl in pairs:
            if img == image_path:
                return lbl
        return None


# Test function
if __name__ == "__main__":
    mapper = IlastikDataMapper()
    
    print("\n" + "="*80)
    print("TESTING DATA MAPPER")
    print("="*80)
    
    for modality in ['AFM', 'CRYO-SEM', 'STED', 'CONFOCAL']:
        print(f"\n{'='*80}")
        print(f"Testing {modality}")
        print(f"{'='*80}")
        pairs = mapper.load_data_pairs(modality)
        print(f"Total pairs found: {len(pairs)}")
        
        if pairs:
            print(f"\nFirst 3 pairs:")
            for i, (img, lbl) in enumerate(pairs[:3]):
                print(f"\n[{i+1}]")
                print(f"  Image: {Path(img).name}")
                print(f"  Label: {Path(lbl).name}")
                print(f"  Image exists: {os.path.exists(img)}")
                print(f"  Label exists: {os.path.exists(lbl)}")