import scipy.ndimage as nd
from pyrp import RP

DEMO_PARAMS = './rp.npy'

if __name__ == '__main__':
    # Load image
    img = nd.imread('./test_images/000013.jpg')

    # Instantiate rp wrapper class
    rp = RP()

    # Load demo parameters
    rp.loadParamsFromNumpy(DEMO_PARAMS)

    # Get the boxes
    boxes = rp.getProposals(img)

    print boxes
