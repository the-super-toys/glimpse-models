from PythonAPI.salicon.salicon import SALICON
import scipy.misc

# The directory has to exist before running this task
output_heatmaps = '../dataset/heatmaps/'
input_fixations_train = 'annotations/fixations_train2014.json'
input_fixations_val = 'annotations/fixations_val2014.json'


"""Supplying the path to either the train or val annotations generates their associated headmap
"""
def generate_heatmap(path):
    salicon = SALICON(path)

    print("getting image IDs")
    imgIds = salicon.getImgIds()

    for imgId in imgIds:
        img = salicon.loadImgs(imgId)[0]
        annIds = salicon.getAnnIds(imgIds=img['id'])
        anns = salicon.loadAnns(annIds)
        heatmap = salicon.buildFixMap(anns)

        print("saving heatmap for " + img['file_name'] + " " + str(heatmap.shape))

        scipy.misc.imsave(output_heatmaps + img['file_name'], heatmap)


print('train')
generate_heatmap(input_fixations_train)

print('valid')
generate_heatmap(input_fixations_val)
