import matplotlib.pyplot as plt

# num is the number of the comparison figures to output
def image_comparison(img, true, pred,num):
    fig, ax = plt.subplots(1, 3, figsize=(20, 16))

    ax[0].imshow(img)
    ax[0].set_title("Seep Image")

    ax[1].imshow(true)
    ax[1].set_title("Mask Image")

    ax[2].imshow(pred)
    ax[2].set_title("Pred Image")
    plt.savefig('output{}.png'.format(num),bbox_inches='tight')
    plt.show()
