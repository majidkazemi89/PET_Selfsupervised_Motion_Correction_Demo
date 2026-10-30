IMPORTANT SCIENTIFIC CONVENTION
--------------------------------
The Interfile header describes the volume as matrix [1..3] = (X,Y,Z) =
(144,144,355), but the raw binary stream is explicitly interpreted here in
storage order (Z,Y,X), with X varying fastest. Therefore the raw bytes are
read as:
    data.reshape(Z, Y, X)
which directly produces the PyTorch spatial order (D,H,W) = (Z,Y,X).
No additional transpose is applied after this raw reshape.

This distinction is intentional and is checked in the loader. The Interfile
matrix sizes describe physical matrix axes; they do not by themselves imply
that NumPy should reshape the byte stream as (X,Y,Z).

Flow channels are always:
    flow[:, 0] = u_x  (W / X direction), voxels
    flow[:, 1] = u_y  (H / Y direction), voxels
    flow[:, 2] = u_z  (D / Z direction), voxels

The network predicts a BACKWARD/SAMPLING deformation vector field (DVF) defined on the
REFERENCE grid:
    warped_moving(x_ref) = moving(x_ref + flow_backward(x_ref))

Therefore the predicted flow is directly suitable for grid_sample().
If an XCAT forward DVF is supplied (reference -> moving), this script can
numerically invert it to obtain the backward/reference-grid field before
comparison.

NO CONVENTIONAL REGISTRATION IS USED BY DEFAULT.
The script never calls image registration to manufacture "ground truth".
True XCAT DVFs must come from the XCAT vector/DVF output generated with the
same XCAT simulation parameters. The PET .i images alone do not contain the
original voxel trajectories, so exact XCAT ground-truth MVFs cannot be
reconstructed from the PET images after the fact.

Two example Dataset_Demo (p80 as training data and p150 as testing data available at a specific activity level of 82 Rb in the torso section):
    p80_ph1_fr36.i ... p80_ph40_fr36.i   -> train/validation
    p150_ph1_fr36.i ... p150_ph40_fr36.i -> independent test

Split:
    p80 cycles 1-4 = phases 1-32 -> training
    p80 cycle 5    = phases 33-40 -> validation
    p150 phases 1-40                -> test

Eight cardiac gates:
    gate k = [k, k+8, k+16, k+24, k+32], k=1..8

Selected Spatial crop:
    Z = Python slice(99, 259), i.e. original slices 99..258, 160 slices.

Voxel spacing:
    X = 2.0000 mm
    Y = 2.0000 mm
    Z = 1.9966 mm

The script is intentionally conservative:
- no phase-label supervision
- no artificial sinusoidal "ground-truth" waveform
- no random pair split
- no silent all-ones myocardium-mask fallback
- no conventional-registration pseudo-ground-truth unless explicitly requested
- validation is cycle-level
- one rolling latest checkpoint is overwritten each epoch for Colab resume
- best validation and final checkpoints are retained
- conventional registration is optional and is NEVER used as ground truth.
