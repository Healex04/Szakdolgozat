import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_modalities(patient_id, nifti_dir, slice_idx=100):
    modalities = ['FLAIR', 'T1', 'T1ce', 'T2']
    suffix = ['0000', '0001', '0002', '0003']
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.1)

    for i, (name, sfx) in enumerate(zip(modalities, suffix)):
        path = os.path.join(nifti_dir, f"{patient_id}_{sfx}.nii.gz")
        if not os.path.exists(path): continue
            
        img = nib.load(path).get_fdata()
        slice_data = np.rot90(img[:, :, slice_idx])
        
        axes[i].imshow(slice_data, cmap='gray')
        axes[i].set_title(name, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f"MRI Modalitások összevetése ({patient_id})", fontsize=16)
    plt.show()

if __name__ == "__main__":
    plot_modalities('BRATS_011', r'C:\nnunet_data\temp_nnunet_in')