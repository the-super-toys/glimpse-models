from PythonAPI.salicon.salicon import SALICON
import scipy.misc

# initialize salicon dataset
salicon = SALICON("../annotations/fixations_train2014.json")

print("getting image IDs")
imgIds = salicon.getImgIds()

for imgId in imgIds:
    img = salicon.loadImgs(imgId)[0]
    annIds = salicon.getAnnIds(imgIds=img['id'])
    anns = salicon.loadAnns(annIds)
    heatmap = salicon.buildFixMap(anns)

    print("saving heatmap for " + img['file_name'] + " " + str(heatmap.shape))

    scipy.misc.imsave('../heatmaps/' + img['file_name'], heatmap)

    # df = pd.DataFrame(heatmap)
    # df.to_csv('../heatmaps/' + img['file_name'].replace(".jpg", ".csv"), header=None, index=None)
