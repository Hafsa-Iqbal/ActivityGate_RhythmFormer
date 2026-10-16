import dataset.data_loader.BaseLoader
import dataset.data_loader.COHFACELoader
import dataset.data_loader.UBFCrPPGLoader
import dataset.data_loader.PURELoader
import dataset.data_loader.SCAMPSLoader
import dataset.data_loader.MMPDLoader
import dataset.data_loader.BP4DPlusLoader
import dataset.data_loader.BP4DPlusBigSmallLoader
import dataset.data_loader.UBFCPHYSLoader
import dataset.data_loader.VIPLHRLoader
import dataset.data_loader.iBVPLoader
import dataset.data_loader.iBVP_RGBThermalLoader
import dataset.data_loader.SIM1Loader
import dataset.data_loader.SIM1RGBTLoader
import dataset.data_loader.SIM1RGBLoader

from .iBVP_RGBThermalLoader import iBVP_RGBThermalLoader


from .iBVP_RGBThermalLoader import iBVP_RGBThermalLoader

# Add to loader registry dict:
dataset_loader_dict = {
    # ... existing loaders
    "iBVP_RGBThermal": iBVP_RGBThermalLoader,
}

