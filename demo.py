import matplotlib.pyplot as plt
import numpy as np
from keras import Model

from yunzhe_model import get_unet_denseblock_x2_deeper


def pcc(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    numerator = np.mean((patch1 - mean1) * (patch2 - mean2))
    denominator = np.sqrt(np.mean(np.square((patch1 - mean1)))) * np.sqrt(np.mean(np.square((patch2 - mean2))))
    return numerator / denominator


def batch_average_pcc(batch):
    num_batches = batch.shape[0]
    pcc_list = np.array([])
    for i in range(num_batches - 1):
        for j in range(i + 1, num_batches, 1):
            pcc_list = np.append(pcc_list, pcc(batch[i, :].squeeze(), batch[j, :].squeeze()))
    return pcc_list.mean()


model = get_unet_denseblock_x2_deeper()
model.load_weights('deep_speckle/mixed2400.hdf5')

intermediate_model1 = Model(inputs=model.inputs, outputs=model.get_layer('max_pooling2d_1').output)
intermediate_model2 = Model(inputs=model.inputs, outputs=model.get_layer('max_pooling2d_2').output)
intermediate_model3 = Model(inputs=model.inputs, outputs=model.get_layer('max_pooling2d_3').output)
intermediate_model4 = Model(inputs=model.inputs, outputs=model.get_layer('max_pooling2d_4').output)
intermediate_model5 = Model(inputs=model.inputs, outputs=model.get_layer('conv2d_27').output)
intermediate_model6 = Model(inputs=model.inputs, outputs=model.get_layer('conv2d_32').output)
intermediate_model7 = Model(inputs=model.inputs, outputs=model.get_layer('conv2d_37').output)
intermediate_model8 = Model(inputs=model.inputs, outputs=model.get_layer('conv2d_42').output)
intermediate_model9 = Model(inputs=model.inputs, outputs=model.get_layer('conv2d_46').output)

data_e = np.load('deep_speckle/speckles/letterE_39.npy')
data_s = np.load('deep_speckle/speckles/letterS_16.npy')
data_8 = np.load('deep_speckle/speckles/number8_56.npy')
data_9 = np.load('deep_speckle/speckles/number9_58.npy')

pred_e = model.predict(data_e)
pred_s = model.predict(data_s)
pred_8 = model.predict(data_8)
pred_9 = model.predict(data_9)

plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(pred_e[i, :, :, 0].squeeze(), cmap='gray', vmin=0, vmax=1)
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(pred_s[i, :, :, 0].squeeze(), cmap='gray', vmin=0, vmax=1)
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(pred_8[i, :, :, 0].squeeze(), cmap='gray', vmin=0, vmax=1)
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(pred_9[i, :, :, 0].squeeze(), cmap='gray', vmin=0, vmax=1)

intermediate_output1_e = intermediate_model1.predict(data_e)
intermediate_output2_e = intermediate_model2.predict(data_e)
intermediate_output3_e = intermediate_model3.predict(data_e)
intermediate_output4_e = intermediate_model4.predict(data_e)
intermediate_output5_e = intermediate_model5.predict(data_e)
intermediate_output6_e = intermediate_model6.predict(data_e)
intermediate_output7_e = intermediate_model7.predict(data_e)
intermediate_output8_e = intermediate_model8.predict(data_e)
intermediate_output9_e = intermediate_model9.predict(data_e)

intermediate_output1_s = intermediate_model1.predict(data_s)
intermediate_output2_s = intermediate_model2.predict(data_s)
intermediate_output3_s = intermediate_model3.predict(data_s)
intermediate_output4_s = intermediate_model4.predict(data_s)
intermediate_output5_s = intermediate_model5.predict(data_s)
intermediate_output6_s = intermediate_model6.predict(data_s)
intermediate_output7_s = intermediate_model7.predict(data_s)
intermediate_output8_s = intermediate_model8.predict(data_s)
intermediate_output9_s = intermediate_model9.predict(data_s)

intermediate_output1_8 = intermediate_model1.predict(data_8)
intermediate_output2_8 = intermediate_model2.predict(data_8)
intermediate_output3_8 = intermediate_model3.predict(data_8)
intermediate_output4_8 = intermediate_model4.predict(data_8)
intermediate_output5_8 = intermediate_model5.predict(data_8)
intermediate_output6_8 = intermediate_model6.predict(data_8)
intermediate_output7_8 = intermediate_model7.predict(data_8)
intermediate_output8_8 = intermediate_model8.predict(data_8)
intermediate_output9_8 = intermediate_model9.predict(data_8)

intermediate_output1_9 = intermediate_model1.predict(data_9)
intermediate_output2_9 = intermediate_model2.predict(data_9)
intermediate_output3_9 = intermediate_model3.predict(data_9)
intermediate_output4_9 = intermediate_model4.predict(data_9)
intermediate_output5_9 = intermediate_model5.predict(data_9)
intermediate_output6_9 = intermediate_model6.predict(data_9)
intermediate_output7_9 = intermediate_model7.predict(data_9)
intermediate_output8_9 = intermediate_model8.predict(data_9)
intermediate_output9_9 = intermediate_model9.predict(data_9)

average_pcc_layers = np.ndarray((4, 10))

average_pcc_layers[0, 0] = batch_average_pcc(data_e)
# average_pcc_layers[0, 1] = batch_average_pcc(intermediate_output1_e)
average_pcc_layers[0, 1] = batch_average_pcc(intermediate_output2_e)
average_pcc_layers[0, 2] = batch_average_pcc(intermediate_output3_e)
average_pcc_layers[0, 3] = batch_average_pcc(intermediate_output4_e)
average_pcc_layers[0, 4] = batch_average_pcc(intermediate_output5_e)
average_pcc_layers[0, 5] = batch_average_pcc(intermediate_output6_e)
average_pcc_layers[0, 6] = batch_average_pcc(intermediate_output7_e)
average_pcc_layers[0, 7] = batch_average_pcc(intermediate_output8_e)
average_pcc_layers[0, 8] = batch_average_pcc(intermediate_output9_e)
average_pcc_layers[0, 9] = batch_average_pcc(pred_e)

average_pcc_layers[1, 0] = batch_average_pcc(data_s)
# average_pcc_layers[1, 1] = batch_average_pcc(intermediate_output1_s)
average_pcc_layers[1, 1] = batch_average_pcc(intermediate_output2_s)
average_pcc_layers[1, 2] = batch_average_pcc(intermediate_output3_s)
average_pcc_layers[1, 3] = batch_average_pcc(intermediate_output4_s)
average_pcc_layers[1, 4] = batch_average_pcc(intermediate_output5_s)
average_pcc_layers[1, 5] = batch_average_pcc(intermediate_output6_s)
average_pcc_layers[1, 6] = batch_average_pcc(intermediate_output7_s)
average_pcc_layers[1, 7] = batch_average_pcc(intermediate_output8_s)
average_pcc_layers[1, 8] = batch_average_pcc(intermediate_output9_s)
average_pcc_layers[1, 9] = batch_average_pcc(pred_s)

average_pcc_layers[2, 0] = batch_average_pcc(data_8)
# average_pcc_layers[2, 1] = batch_average_pcc(intermediate_output1_8)
average_pcc_layers[2, 1] = batch_average_pcc(intermediate_output2_8)
average_pcc_layers[2, 2] = batch_average_pcc(intermediate_output3_8)
average_pcc_layers[2, 3] = batch_average_pcc(intermediate_output4_8)
average_pcc_layers[2, 4] = batch_average_pcc(intermediate_output5_8)
average_pcc_layers[2, 5] = batch_average_pcc(intermediate_output6_8)
average_pcc_layers[2, 6] = batch_average_pcc(intermediate_output7_8)
average_pcc_layers[2, 7] = batch_average_pcc(intermediate_output8_8)
average_pcc_layers[2, 8] = batch_average_pcc(intermediate_output9_8)
average_pcc_layers[2, 9] = batch_average_pcc(pred_8)

average_pcc_layers[3, 0] = batch_average_pcc(data_9)
# average_pcc_layers[3, 1] = batch_average_pcc(intermediate_output1_9)
average_pcc_layers[3, 1] = batch_average_pcc(intermediate_output2_9)
average_pcc_layers[3, 2] = batch_average_pcc(intermediate_output3_9)
average_pcc_layers[3, 3] = batch_average_pcc(intermediate_output4_9)
average_pcc_layers[3, 4] = batch_average_pcc(intermediate_output5_9)
average_pcc_layers[3, 5] = batch_average_pcc(intermediate_output6_9)
average_pcc_layers[3, 6] = batch_average_pcc(intermediate_output7_9)
average_pcc_layers[3, 7] = batch_average_pcc(intermediate_output8_9)
average_pcc_layers[3, 8] = batch_average_pcc(intermediate_output9_9)
average_pcc_layers[3, 9] = batch_average_pcc(pred_9)

plt.figure()
plt.plot(average_pcc_layers[0, :].squeeze(), 'r', label='letter E')
plt.plot(average_pcc_layers[1, :].squeeze(), 'g', label='letter S')
plt.plot(average_pcc_layers[2, :].squeeze(), 'b', label='number 8')
plt.plot(average_pcc_layers[3, :].squeeze(), 'k', label='number 9')
plt.legend(loc='upper left')
plt.ylabel('Pearson Correlation Coefficient')
plt.ylim(0, 1)
plt.title('PCC of intermediate activation map')
plt.xticks(np.arange(10), (
    'Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Prediction'))

channels = [27, 19, 15, 4, 19, 8, 23, 30]

inter_layers = {'layer1': intermediate_output2_e,
                'layer2': intermediate_output3_e,
                'layer3': intermediate_output4_e,
                'layer4': intermediate_output5_e,
                'layer5': intermediate_output6_e,
                'layer6': intermediate_output7_e,
                'layer7': intermediate_output8_e,
                'layer8': intermediate_output9_e}

plt.figure(figsize=[20, 4])
plt.subplot(2, 10, 1)
plt.imshow(data_e[0, :].squeeze(), cmap='hot')
plt.axis('off')
plt.subplot(2, 10, 11)
plt.imshow(data_e[1, :].squeeze(), cmap='hot')
plt.axis('off')
plt.subplot(2, 10, 10)
plt.imshow(pred_e[0, :, :, 0].squeeze(), cmap='gray')
plt.axis('off')
plt.subplot(2, 10, 20)
plt.imshow(pred_e[1, :, :, 0].squeeze(), cmap='gray')
plt.axis('off')


for i in range(4,8):
    data = inter_layers['layer' + str(i + 1)]
    for j in range(data.shape[-1]):
        plt.figure(figsize=[4, 8])
        plt.subplot(2, 1, 1)
        plt.imshow(data[0, :, :, j], cmap='hot')
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(data[1, :, :, j], cmap='hot')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.08, hspace=0.08)
        plt.savefig('deep_speckle/activations/'+str(i+1)+'/'+str(j+1)+'.png')
        plt.close('all')


# for i in range(8):
#     data = inter_layers['layer' + str(i + 1)]
#     plt.subplot(2, 10, i + 2)
#     plt.imshow(data[0, :, :, channels[i]], cmap='hot')
#     plt.axis('off')
#     plt.subplot(2, 10, i + 2 + 10)
#     plt.imshow(data[1, :, :, channels[i]], cmap='hot')
#     plt.axis('off')
#
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.08, hspace=0.08)
