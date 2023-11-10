import numpy as np
import scipy as scp
import pickle
import gzip
import os
import re
import hashlib
import itertools
import warnings
import copy
import inspect

# scipy.optimize.curve_fit() throws an exception when the optimizer exhausts the maximum allowed number of function
# evaluations. Since we are demanding fairly narrow tolerance, reaching the maximum number of evaluations is not
# necessarily fatal (it merely means that the actual tolerance is somewhat less than our default standard). So we would
# like to deescalate the error to a warning and proceed. Unfortunately, I am unable to find a more elegant way of doing
# it except by overriding the optimizer function that curve_fit() calls
original_least_squares = scp.optimize._minpack_py.least_squares
def least_squares_wrapper(*args, **kwargs):
    res = original_least_squares(*args, **kwargs)
    if (not res.success) and (res.status == 0):
        warn('Optimizer failed to reach desired convergence after the maximum number ({}) of function evaluations'.format(res.nfev))
        res.success = True
    return res
scp.optimize._minpack_py.least_squares = least_squares_wrapper

script_dir = os.path.dirname(os.path.realpath(__file__))

warnings_stack = []
warnings_messages = {}

settings = {
    ### Model grid settings ###
    'griddir': script_dir + '/../evan_models/',    # Path to the grid directory (must have grid7/bin and gridie/bin subdirectories)

    ### Spectrograph settings ###
    'arms': {                                      # Parameters of individual spectrograph arms
        'blue': {
            'FWHM': 2.07,                               # FWHM of the line spread function in the arm
            'wl': np.linspace(3800, 6500, 4096),       # "Typical" bin wavelengths in the arm (exact wavelengths will vary between
                                                       # observations due to changing wavelength calibration)
            'priority': 1,                             # If multiple arms have overlapping wavelength ranges, the arm with the
                                                       # highest priority will be used in the overlap region
        },
        'red_lr': {
            'FWHM': 2.63,
            'wl': np.linspace(6300, 9700, 4096),
            'priority': 2,
        },
        'red_mr': {
            'FWHM': 1.368,
            'wl': np.linspace(7100, 8850, 4096),
        },
        'ir': {
            'FWHM': 2.4,
            'wl': np.linspace(9400, 12600, 4096),
        }
    },

    ### Fitting masks ###
    'masks': {
        'red_mr': {
            'alpha': [[6317.545, 6321.895], [7690.26, 7692.87], [7810.32, 7811.915], [7875.86, 7878.18], [7895.0, 7897.61], [8097.71, 8099.74], [8211.68, 8215.16], [8344.79, 8347.4],
                      [8711.64, 8713.67], [8716.715, 8718.89], [8734.695, 8737.45], [8922.47, 8924.645], [6345.675, 6348.575], [7002.525, 7007.02], [7016.59, 7018.475], [7033.845, 7036.02],
                      [7163.765, 7166.81], [7192.185, 7194.94], [7234.38, 7236.7], [7249.46, 7251.78], [7274.255, 7276.43], [7281.795, 7283.825], [7287.885, 7290.785], [7404.465, 7407.075],
                      [7407.8, 7410.265], [7414.47, 7417.37], [7422.01, 7425.2], [7678.95, 7681.56], [7741.445, 7744.055], [7798.865, 7800.895], [7849.035, 7850.92], [7917.185, 7919.505],
                      [7931.105, 7933.715], [7942.705, 7945.315], [8500.665, 8503.275], [8535.32, 8537.06], [8555.33, 8558.23], [8647.115, 8649.725], [8726.865, 8729.475], [8741.075, 8743.83],
                      [8750.355, 8753.4], [8789.215, 8791.535], [8891.585, 8893.905], [6437.17, 6441.085], [6448.19, 6451.525], [6453.99, 6457.905], [6460.66, 6466.025], [6469.94, 6473.275],
                      [6491.835, 6495.75], [6497.925, 6501.405], [6507.64, 6510.105], [6571.15, 6574.485], [6715.86, 6719.485], [7146.22, 7150.135], [7200.45, 7204.075], [7324.425, 7327.76],
                      [8200.37, 8203.125], [8247.35, 8250.25], [8489.355, 8508.785], [8518.645, 8564.175], [8633.34, 8634.645], [8644.215, 8679.16], [8910.435, 8913.625], [8925.805, 8928.995]],
            'zscale': [[6300.0, 6304.35], [6309.86, 6313.195], [6313.63, 6319.72], [6321.025, 6324.36], [6329.725, 6332.045], [6333.64, 6340.455], [6342.485, 6345.675], [6352.49, 6360.32],
                       [6361.915, 6365.83], [6368.15, 6370.905], [6379.315, 6382.505], [6382.65, 6386.42], [6391.06, 6395.41], [6398.165, 6402.08], [6406.14, 6413.39], [6415.42, 6423.105],
                       [6429.05, 6434.27], [6435.43, 6437.46], [6454.715, 6457.905], [6461.24, 6464.285], [6467.475, 6470.665], [6474.0, 6477.19], [6480.235, 6485.31], [6491.11, 6503.0],
                       [6514.6, 6519.965], [6527.65, 6529.535], [6532.725, 6535.19], [6544.615, 6548.675], [6550.27, 6553.17], [6567.815, 6570.86], [6572.6, 6576.805], [6579.705, 6582.75],
                       [6590.58, 6595.655], [6596.235, 6598.845], [6606.53, 6611.17], [6612.33, 6615.375], [6623.495, 6628.715], [6632.05, 6635.385], [6638.575, 6641.04], [6645.68, 6649.45],
                       [6652.93, 6654.815], [6661.63, 6668.735], [6676.275, 6679.755], [6695.56, 6697.155], [6702.085, 6706.725], [6708.755, 6718.47], [6723.98, 6730.215], [6731.375, 6734.275],
                       [6736.015, 6741.09], [6744.425, 6757.04], [6761.535, 6763.275], [6782.125, 6788.07], [6792.13, 6797.06], [6800.685, 6811.56], [6818.665, 6822.87], [6827.22, 6829.975],
                       [6836.065, 6845.2], [6850.275, 6864.63], [6874.055, 6877.1], [6879.42, 6882.465], [6884.64, 6886.96], [6897.255, 6899.43], [6910.16, 6913.06], [6915.235, 6918.28],
                       [6929.88, 6935.1], [6943.51, 6948.585], [6949.6, 6953.805], [6959.46, 6961.2], [6969.61, 6973.235], [6974.685, 6980.775], [6987.01, 6990.055], [6998.465, 7001.655],
                       [7005.135, 7018.04], [7021.375, 7026.015], [7026.885, 7032.54], [7036.745, 7039.79], [7043.705, 7045.59], [7057.19, 7059.075], [7066.47, 7073.43], [7082.42, 7084.45],
                       [7085.32, 7088.075], [7088.945, 7096.63], [7101.995, 7104.315], [7106.2, 7108.665], [7110.695, 7115.915], [7126.5, 7134.475], [7141.435, 7143.9], [7144.19, 7146.22],
                       [7147.815, 7152.89], [7154.63, 7159.705], [7161.735, 7166.085], [7174.93, 7182.76], [7185.66, 7193.49], [7193.78, 7196.39], [7205.525, 7209.15], [7211.325, 7214.225],
                       [7218.14, 7226.55], [7227.275, 7230.03], [7238.44, 7241.34], [7244.385, 7245.4], [7252.505, 7257.0], [7257.87, 7262.8], [7267.73, 7269.47], [7277.445, 7279.765],
                       [7281.65, 7286.435], [7287.015, 7290.205], [7291.365, 7294.99], [7299.485, 7302.675], [7305.14, 7312.97], [7314.855, 7318.19], [7319.35, 7325.44], [7332.4, 7336.17],
                       [7342.84, 7354.585], [7362.85, 7367.635], [7368.94, 7370.97], [7380.975, 7384.165], [7384.89, 7391.125], [7399.68, 7403.16], [7409.685, 7412.73], [7417.08, 7422.59],
                       [7428.97, 7432.16], [7434.625, 7438.105], [7439.41, 7450.575], [7452.895, 7455.215], [7459.855, 7464.64], [7470.875, 7483.49], [7490.16, 7496.83], [7497.265, 7499.73],
                       [7504.95, 7508.72], [7509.3, 7513.07], [7513.65, 7517.13], [7521.77, 7524.09], [7529.6, 7534.53], [7539.17, 7542.65], [7544.825, 7548.74], [7558.745, 7560.775],
                       [7562.225, 7563.82], [7567.445, 7570.345], [7581.51, 7590.21], [7592.82, 7595.14], [7604.565, 7607.03], [7609.495, 7611.09], [7614.135, 7621.965], [7649.515, 7657.055],
                       [7659.665, 7666.045], [7708.82, 7713.315], [7713.605, 7715.635], [7717.955, 7720.42], [7721.58, 7724.77], [7744.49, 7752.32], [7779.0, 7782.19], [7806.55, 7809.45],
                       [7809.74, 7811.77], [7830.62, 7833.81], [7854.4, 7856.43], [7868.465, 7870.785], [7911.24, 7914.575], [7935.455, 7938.79], [7939.515, 7942.995], [7944.155, 7947.49],
                       [7953.725, 7956.77], [7958.22, 7960.25], [7993.165, 7996.065], [7996.935, 8000.56], [8025.79, 8029.85], [8042.175, 8049.57], [8070.885, 8076.83], [8079.15, 8081.905],
                       [8083.645, 8086.69], [8095.535, 8098.58], [8106.7, 8109.6], [8110.76, 8113.515], [8128.16, 8130.335], [8144.545, 8147.445], [8148.46, 8150.78], [8178.04, 8179.925],
                       [8185.725, 8188.045], [8196.02, 8200.37], [8202.545, 8209.215], [8218.64, 8222.12], [8230.675, 8233.865], [8237.49, 8240.68], [8246.625, 8249.67], [8258.66, 8260.835],
                       [8263.3, 8264.895], [8268.23, 8270.55], [8274.755, 8277.365], [8286.79, 8290.56], [8291.865, 8295.055], [8302.16, 8304.045], [8305.06, 8312.31], [8325.215, 8329.42],
                       [8330.145, 8333.625], [8337.83, 8346.24], [8347.545, 8350.59], [8357.26, 8362.19], [8364.22, 8367.12], [8380.75, 8383.65], [8385.825, 8389.74], [8394.235, 8396.12],
                       [8400.035, 8402.935], [8421.35, 8427.15], [8433.385, 8435.125], [8438.17, 8441.36], [8445.275, 8449.045], [8450.35, 8451.655], [8465.43, 8472.825], [8476.305, 8477.9],
                       [8479.93, 8483.555], [8495.59, 8498.49], [8512.41, 8517.195], [8524.59, 8528.505], [8537.06, 8539.235], [8560.84, 8563.015], [8570.845, 8572.875], [8580.705, 8584.185],
                       [8591.58, 8594.19], [8597.38, 8600.28], [8606.225, 8608.4], [8609.415, 8617.535], [8620.0, 8623.335], [8631.455, 8633.34], [8651.755, 8655.67], [8660.455, 8664.37],
                       [8666.255, 8668.865], [8671.185, 8676.55], [8677.855, 8681.335], [8686.555, 8691.05], [8697.285, 8701.055], [8708.885, 8714.685], [8727.01, 8730.49], [8746.005, 8748.76],
                       [8755.43, 8758.91], [8762.39, 8765.58], [8783.415, 8785.59], [8789.215, 8798.35], [8822.275, 8826.19], [8832.57, 8834.745], [8836.775, 8840.255], [8845.04, 8848.23],
                       [8861.28, 8864.325], [8865.34, 8869.98], [8874.765, 8880.275], [8905.07, 8906.955], [8918.7, 8923.34], [8925.66, 8927.69], [8927.98, 8933.055], [8941.32, 8947.845],
                       [8949.15, 8951.18], [8973.8, 8979.165], [8983.66, 8985.98], [8993.665, 8995.695], [8997.87, 9001.35], [9006.28, 9015.56], [9018.75, 9020.635], [9022.955, 9026.0],
                       [9029.19, 9032.38], [9057.175, 9058.915], [9061.09, 9063.41], [9069.21, 9071.53], [9078.055, 9081.68], [9083.42, 9085.015], [9086.32, 9091.25], [9097.63, 9099.515]],
        },
        'red_lr': {
            'alpha': [[6309.86, 6312.76], [7689.68, 7693.45], [7810.465, 7811.915], [7875.425, 7878.615], [7894.42, 7898.335], [8097.275, 8100.175], [8210.955, 8215.595], [8344.355, 8347.98],
                      [8710.625, 8713.96], [8716.28, 8719.325], [8733.97, 8738.03], [8922.035, 8925.08], [6369.31, 6373.515], [7002.235, 7007.31], [7016.3, 7018.62], [7033.41, 7036.455],
                      [7163.185, 7167.245], [7191.605, 7195.665], [7249.025, 7252.215], [7273.82, 7276.72], [7281.65, 7284.115], [7287.16, 7291.365], [7403.885, 7410.845], [7413.745, 7417.95],
                      [7421.285, 7425.925], [7668.655, 7670.54], [7740.865, 7744.635], [7798.72, 7801.185], [7848.89, 7851.21], [7916.75, 7920.085], [7930.38, 7934.295], [7941.98, 7946.04],
                      [8443.535, 8444.405], [8500.23, 8503.71], [8535.175, 8537.205], [8554.75, 8558.955], [8646.39, 8650.305], [8685.54, 8687.28], [8726.285, 8730.055], [8740.495, 8744.41],
                      [8749.63, 8754.125], [8788.925, 8792.26], [6436.155, 6441.955], [6447.175, 6452.685], [6452.975, 6458.34], [6459.645, 6466.75], [6469.07, 6474.29], [6490.82, 6502.275],
                      [6507.205, 6510.54], [6570.135, 6575.355], [6714.99, 6720.355], [7145.205, 7151.005], [7199.435, 7204.945], [7323.555, 7328.775], [8199.645, 8203.85], [8246.48, 8251.12],
                      [8252.86, 8256.63], [8489.355, 8508.785], [8518.645, 8564.175], [8644.07, 8679.45], [8909.565, 8914.64], [8924.935, 8929.865], [9097.195, 9099.805]],
            'zscale': [[6300.0, 6305.51], [6307.25, 6325.23], [6329.145, 6346.545], [6351.475, 6366.555], [6367.28, 6371.63], [6378.59, 6386.71], [6390.19, 6396.57], [6397.005, 6403.095],
                       [6405.415, 6423.975], [6428.035, 6438.62], [6453.99, 6458.775], [6460.515, 6465.3], [6466.46, 6471.535], [6473.13, 6478.205], [6479.365, 6485.89], [6491.4, 6503.725],
                       [6513.73, 6520.835], [6527.36, 6529.825], [6532.0, 6535.915], [6543.745, 6554.04], [6556.07, 6557.375], [6566.945, 6577.82], [6578.69, 6583.91], [6590.0, 6599.425],
                       [6605.66, 6616.1], [6622.625, 6629.15], [6631.18, 6636.4], [6637.995, 6641.62], [6644.955, 6650.32], [6652.64, 6654.96], [6660.615, 6669.46], [6675.26, 6680.77],
                       [6695.415, 6698.17], [6701.07, 6718.76], [6723.545, 6741.815], [6744.135, 6757.475], [6761.535, 6763.42], [6781.4, 6788.94], [6791.55, 6797.205], [6800.25, 6812.285],
                       [6818.085, 6823.16], [6826.35, 6830.845], [6835.485, 6846.215], [6849.405, 6864.92], [6873.33, 6877.825], [6878.985, 6887.395], [6896.53, 6899.865], [6909.435, 6919.44],
                       [6929.735, 6936.405], [6942.495, 6954.095], [6959.46, 6961.345], [6969.32, 6981.935], [6986.14, 6990.925], [6997.45, 7002.525], [7005.135, 7019.055], [7020.505, 7032.975],
                       [7035.875, 7040.66], [7043.415, 7045.88], [7056.9, 7059.22], [7065.745, 7073.72], [7082.13, 7097.21], [7100.69, 7120.265], [7125.34, 7135.345], [7140.855, 7160.285],
                       [7161.3, 7166.955], [7174.35, 7183.63], [7184.79, 7196.97], [7204.51, 7230.755], [7237.715, 7242.065], [7252.65, 7263.67], [7267.44, 7269.47], [7276.865, 7280.055],
                       [7281.65, 7295.715], [7299.05, 7326.165], [7331.53, 7336.025], [7342.26, 7354.875], [7362.56, 7371.115], [7375.175, 7377.205], [7380.395, 7391.995], [7395.91, 7396.925],
                       [7398.81, 7403.885], [7408.815, 7413.6], [7416.355, 7422.88], [7427.375, 7433.175], [7434.335, 7451.3], [7452.46, 7455.36], [7459.13, 7465.075], [7470.44, 7487.26],
                       [7489.29, 7500.89], [7504.225, 7517.71], [7521.335, 7524.38], [7528.585, 7535.11], [7538.3, 7543.085], [7543.955, 7549.32], [7558.455, 7563.82], [7566.575, 7571.215],
                       [7580.93, 7590.355], [7592.24, 7595.575], [7604.275, 7607.32], [7609.495, 7611.09], [7613.7, 7622.545], [7658.65, 7666.915], [7708.095, 7715.925], [7717.52, 7725.64],
                       [7733.18, 7734.195], [7744.055, 7752.9], [7778.13, 7783.06], [7805.825, 7811.915], [7829.75, 7834.68], [7843.815, 7845.12], [7867.885, 7871.22], [7910.37, 7915.59],
                       [7934.585, 7948.36], [7953.145, 7960.54], [7992.44, 8002.155], [8024.63, 8030.72], [8041.885, 8050.295], [8070.305, 8087.56], [8089.445, 8090.75], [8094.955, 8099.015],
                       [8106.12, 8114.24], [8127.725, 8130.915], [8144.11, 8151.215], [8177.895, 8180.07], [8185.29, 8188.48], [8196.02, 8210.085], [8217.625, 8222.99], [8229.805, 8235.315],
                       [8236.04, 8243.0], [8245.9, 8250.395], [8258.225, 8261.27], [8263.3, 8264.895], [8267.94, 8270.84], [8274.32, 8277.945], [8286.5, 8296.07], [8301.725, 8304.335],
                       [8305.495, 8312.89], [8324.2, 8334.64], [8336.67, 8351.46], [8356.535, 8367.845], [8380.025, 8390.61], [8393.945, 8396.265], [8399.165, 8403.805], [8413.52, 8415.405],
                       [8420.625, 8427.585], [8433.24, 8435.27], [8437.3, 8441.94], [8444.695, 8449.915], [8464.995, 8473.26], [8476.305, 8477.9], [8479.64, 8483.845], [8492.545, 8499.795],
                       [8511.395, 8518.21], [8524.3, 8529.085], [8536.77, 8539.525], [8560.26, 8563.305], [8570.41, 8573.165], [8579.835, 8585.78], [8591.145, 8594.625], [8596.655, 8601.005],
                       [8606.08, 8624.205], [8631.31, 8633.775], [8651.755, 8656.54], [8659.875, 8664.66], [8665.675, 8669.155], [8671.185, 8681.915], [8685.54, 8691.92], [8696.56, 8701.78],
                       [8708.16, 8715.555], [8726.575, 8731.07], [8745.28, 8749.63], [8754.56, 8759.78], [8761.375, 8766.305], [8782.98, 8785.88], [8788.635, 8798.495], [8821.26, 8827.35],
                       [8832.425, 8841.125], [8844.46, 8848.955], [8861.135, 8870.85], [8874.33, 8881.0], [8896.805, 8898.255], [8904.635, 8907.245], [8917.975, 8923.195], [8925.37, 8933.78],
                       [8940.16, 8951.615], [8972.785, 8979.6], [8983.37, 8986.56], [8993.52, 9002.365], [9006.135, 9016.43], [9018.605, 9021.07], [9022.085, 9026.725], [9028.465, 9033.105],
                       [9057.175, 9059.06], [9060.51, 9063.845], [9068.775, 9071.965], [9077.33, 9092.41], [9097.485, 9099.805]],
        },
        'blue': {
            'alpha': [[4164.235, 4169.31], [4350.27, 4355.055], [4379.415, 4381.3], [4700.59, 4705.375], [5152.845, 5202.435], [5525.93, 5530.86], [5709.21, 5712.98], [5783.885, 5786.64],
                      [4100.725, 4105.075], [4126.68, 4132.48], [4429.875, 4432.195], [4782.37, 4783.675], [4791.36, 4793.39], [4866.325, 4867.92], [5039.89, 5042.21], [5054.39, 5057.58],
                      [5664.405, 5667.305], [5674.7, 5676.295], [5682.965, 5686.01], [5689.2, 5691.665], [5699.93, 5702.25], [5706.89, 5709.935], [5747.055, 5748.36], [5752.565, 5754.885],
                      [5770.835, 5773.445], [5779.535, 5781.275], [5791.86, 5794.325], [5796.645, 5798.965], [5947.01, 5950.2], [5957.015, 5958.175], [6141.745, 6146.095], [6153.635, 6156.825],
                      [6235.995, 6238.75], [6242.52, 6245.71], [6253.54, 6255.135], [4106.38, 4110.585], [4131.175, 4134.075], [4135.38, 4142.05], [4145.53, 4271.39], [4272.26, 4294.155],
                      [4294.735, 4309.525], [4310.685, 4314.02], [4315.47, 4321.705], [4353.17, 4357.085], [4422.77, 4427.99], [4432.195, 4438.14], [4451.915, 4458.73], [4471.055, 4472.94],
                      [4510.495, 4513.975], [4524.705, 4528.91], [4576.47, 4588.07], [4683.48, 4687.105], [4719.73, 4722.34], [4799.48, 4800.495], [4845.88, 4848.78], [4876.185, 4879.955],
                      [5000.305, 5002.77], [5018.575, 5021.765], [5039.6, 5043.515], [5258.55, 5267.54], [5268.7, 5272.035], [5306.545, 5307.995], [5338.445, 5339.895], [5347.435, 5351.64],
                      [5510.995, 5514.91], [5580.16, 5583.785], [5586.54, 5604.81], [5855.225, 5859.72], [5865.955, 5869.145], [6096.36, 6098.1], [6100.275, 6105.205], [6119.56, 6124.925],
                      [6154.505, 6157.55], [6159.145, 6171.76]],
            'zscale': [[4100.0, 4355.2], [4356.36, 4696.965], [4697.4, 4722.775], [4724.37, 4752.645], [4754.24, 4897.935], [4900.11, 5033.365], [5034.525, 5093.25], [5094.845, 5156.035],
                       [5157.195, 5174.595], [5175.465, 5181.845], [5182.57, 5238.685], [5240.28, 5310.025], [5312.635, 5335.4], [5336.27, 5343.375], [5348.015, 5351.495], [5351.93, 5354.685],
                       [5356.86, 5418.63], [5420.225, 5458.215], [5459.23, 5509.11], [5510.415, 5514.185], [5515.055, 5605.39], [5606.405, 5628.88], [5631.925, 5669.625], [5676.73, 5681.37],
                       [5684.415, 5688.475], [5689.925, 5725.015], [5729.8, 5744.3], [5746.475, 5756.19], [5758.075, 5765.035], [5768.515, 5787.365], [5789.54, 5799.835], [5802.735, 5818.395],
                       [5826.08, 5829.415], [5832.46, 5839.855], [5844.785, 5865.085], [5871.03, 5874.51], [5875.96, 5885.675], [5890.17, 5894.665], [5897.42, 5918.445], [5926.13, 5936.57],
                       [5939.615, 5945.27], [5946.865, 5960.64], [5961.655, 5964.845], [5973.545, 5978.91], [5981.81, 5988.915], [5989.64, 5993.41], [5995.73, 6000.08], [6000.95, 6028.935],
                       [6033.14, 6037.2], [6040.39, 6043.725], [6053.295, 6058.08], [6059.53, 6067.65], [6076.64, 6086.935], [6087.95, 6106.22], [6111.875, 6116.805], [6118.4, 6122.17],
                       [6126.085, 6130.725], [6134.205, 6140.295], [6141.02, 6142.325], [6144.935, 6153.78], [6155.955, 6167.12], [6168.715, 6175.53], [6178.285, 6193.8], [6197.86, 6202.355],
                       [6211.2, 6222.655], [6224.975, 6234.545], [6236.575, 6242.665], [6244.26, 6249.77], [6250.35, 6257.89], [6262.965, 6273.26], [6276.015, 6282.975], [6288.92, 6294.14],
                       [6295.59, 6299.795]],
        },
        'all': {
            'all': [[4500., 5164.322], [5170.322, 5892.924], [5898.924, 8488.023], [8508.023, 8525.091], [8561.091, 8645.141], [8679.141, 9100.]],
        },
        'continuum': [[6864, 6935], [7591, 7694], [8938, 9100]],
    },

    ### Fitting sequence ###
    'continuum': [['zscale', 'alpha', 'teff', 'logg']],
    'redeterminations': [],#['zscale', 'alpha', 'zscale'],

    ### Optimization parameters ###
    'curve_fit': {
        'absolute_sigma': True,
        'ftol': 1e-10,
        'gtol': 1e-10,
        'xtol': 1e-10,
    },
    'cont_pix_refined': 100,
    'spline_order': 3,
    'cont_maxiter': 500,

    ### Continuum iteration thersholds ###
    'thresholds': {
        'teff': 1e-3,
        'zscale': 1e-9,
        'logg': 1e-9,
        'alpha': 1e-9,
    },

    ### Warnings ###
    'throw_python_warnings': True,
}

def warn(message):
    """Issue a warning. Wrapper for `warnings.warn()`
    
    Add the warning message to the stack. If required, also throw the normal Python warning

    Instead of storing warning messages in the stack in full, we associate unique numerical
    identifiers with each distinct message and store them in a dictionary. The stack then
    only contains the identifiers to save memory
    
    Parameters
    ----------
    message : str
        Warning message
    """
    global warnings_stack, warnings_messages

    if message not in warnings_messages:
        warning_id = len(warnings_messages)
        warnings_messages[message] = warning_id
    else:
        warning_id = warnings_messages[message]

    warnings_stack += [warning_id]
    if settings['throw_python_warnings']:
        warnings.warn(message)

def read_grid_model(params):
    """Load a specific model spectrum from the model grid
    
    This version of the function interfaces with Grid7/GridIE.
    `settings['griddir']` must be pointed to the directories that contains the
    "grid7/bin" and "gridie/bin" subdirectories
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters. A value must be provided for each grid
        axis, keyed by the axis name
    
    Returns
    -------
    wl : array_like
        Grid of model wavelengths in A
    flux : array_like
        Corresponding continuum-normalized flux densities
    """
    wl = np.array([], dtype = float)
    flux = np.array([], dtype = float)

    # Load the blue synthetic spectrum, then the red one
    for blue_or_red in [True, False]:

        # Determine the path to the file containing the spectrum
        subgrid_dir = ['grid7', 'gridie'][int(blue_or_red)]
        params_formatted = {}
        for param in ['logg', 'zscale', 'alpha']:
            params_formatted[param] = ['_', '-'][int(params[param] < 0)] + '{:02d}'.format(int(np.round(np.abs(params[param]) * 10)))
        params_formatted['teff'] = int(params['teff'])
        filename = 't{teff}g{logg}f{zscale}a{alpha}.bin.gz'.format(**params_formatted)
        cards = [settings['griddir'], subgrid_dir, 'bin', params_formatted['teff'], params_formatted['logg'], filename]
        if not os.path.isfile(path := (template := '{}/{}/{}/t{}/g{}/{}').format(*cards)):
            cards[-1] = cards[-1].replace('a', 'a_00a')
            if not os.path.isfile(path := template.format(*cards)):
                raise ValueError('Cannot locate {}'.format(path))

        # Load flux from the binary file
        f = gzip.open(path, 'rb')
        file_flux = 1.0 - np.frombuffer(f.read(), dtype = np.float32)
        f.close()

        # Generate the corresponding wavelengths based on some hard-coded parameters
        step = 0.14
        if blue_or_red:
            start = 4100
            stop = 6300
        else:
            start = 6300
            stop = 9100
        file_wl = np.arange(start, stop + step, step)
        file_wl = file_wl[file_wl <= stop + step * 0.1]
        if len(file_wl) != len(file_flux):
            raise ValueError('Unexpected number of points in {}'.format(path))

        wl = np.concatenate([wl, file_wl])
        flux = np.concatenate([flux, file_flux])

    if not np.all(wl[1:] > wl[:-1]):
        raise ValueError('Model wavelengths out of order')
    return wl, flux

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions
    
    This version of the function interfaces with Grid7/GridIE.
    `settings['griddir']` must be pointed to the directories that contains the
    "grid7/bin" and "gridie/bin" subdirectories

    The function implements file caching
    
    Parameters
    ----------
    flush_cache : bool, optional
        If True, discard cache and read the grid afresh
    
    Returns
    -------
    dict
        Dictionary of lists, keyed be grid axis names. The lists contain unique
        values along the corresponding axis that are available in the grid
    """
    # Apply caching
    if os.path.isfile(cache := (settings['griddir'] + '/cache.pkl')) and (not flush_cache):
        f = open(cache, 'rb')
        grid = pickle.load(f)
        f.close()
        return grid

    # Grid7 only has four dimensions: Teff, log(g), [M/H] and [alpha/M]
    grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': []}

    # Recursively collect and parse the filenames of all *.bin.gz models
    for root, subdirs, files in os.walk(settings['griddir']):
        for file in files:
            if file[-7:].lower() == '.bin.gz':
                breakdown = list(re.findall('t([0-9]{4})g([_-][0-9]{2})f([_-][0-9]{2})a([_-][0-9]{2})\.', file.replace('a_00a', 'a'))[0])
                breakdown[0] = int(breakdown[0])
                for i in range(1, 4):
                    breakdown[i] = np.round(float(breakdown[i].replace('_', '')) / 10.0, 1)
                grid['teff'] += [breakdown[0]]
                grid['logg'] += [breakdown[1]]
                grid['zscale'] += [breakdown[2]]
                grid['alpha'] += [breakdown[3]]

    grid = {axis: np.unique(grid[axis]) for axis in grid}
    f = open(cache, 'wb')
    pickle.dump(grid, f)
    f.close()
    return grid

def convolution_integral(sigma, segment_left, segment_right, bin_left, bin_right):
    """Calculate weights of the convolution for an arbitrary flux density spectrum
    and a Gaussian kernel with fixed or slowly-varying standard deviation
    
    A source with a given (true) flux density spectrum is observed with a detector
    of given resolution that records the average flux density in a set of wavelength
    bins. This function estimates the linear coefficients `C1` and `C2` such that

    ``C1 * A + C2 * B``

    gives the contribution of the spectrum segment between the wavelengths
    `segment_left` and `segment_right`, to the detector bin with edge wavelengths
    `bin_left` and `bin_right`. It is assumed that the resolution of the detector
    is constant for each detector bin / spectrum segment pair, and that the flux
    density within the segment varies linearly with wavelength:

    ``flux density = A + B * wavelength``

    This function is primarily designed to handle model spectra, for which the flux
    density has been calculated at given wavelength points, and can be linearly
    interpolated between those points. The weight coefficients may then be evaluated
    for each segment between adjacent wavelength points, and the contributions
    of each segment can be added together to obtain the total flux density in the
    detector bin
    
    Parameters
    ----------
    sigma : array_like
        Detector resolution for the flux arriving from the spectrum segment of
        interest at the detector bin of interest. Adopted as the standard deviation
        of the Gaussian convolution kernel
    segment_left : array_like
        Lower wavelength bound of the spectrum segment of interest
    segment_right : array_like
        Upper wavelength bound of the spectrum segment of interest
    bin_left : array_like
        Lower wavelength bound of the detector bin of interest
    bin_right : array_like
        Upper wavelength bound of the detector bin of interest
    
    Returns
    -------
    C1 : array_like
        Weight coefficient of the vertical offset in the flux density spectrum segment
    C2 : array_like
        Weight coefficient of the linear slope in the flux density spectrum segment
    """
    sqrt_sigma = np.sqrt(2) * sigma
    sqpi = np.sqrt(np.pi)
    sq2pi = np.sqrt(2) * sqpi

    blsl = (segment_left - bin_left) / sqrt_sigma; blsl_sigma = segment_left ** 2 - bin_left ** 2 + sigma ** 2
    blsr = (segment_left - bin_right) / sqrt_sigma; blsr_sigma = segment_left ** 2 - bin_right ** 2 + sigma ** 2
    brsl = (segment_right - bin_left) / sqrt_sigma; brsl_sigma = segment_right ** 2 - bin_left ** 2 + sigma ** 2
    brsr = (segment_right - bin_right) / sqrt_sigma; brsr_sigma = segment_right ** 2 - bin_right ** 2 + sigma ** 2

    erfblsl = scp.special.erf(blsl); expblsl = np.exp(-blsl ** 2)
    erfblsr = scp.special.erf(blsr); expblsr = np.exp(-blsr ** 2)
    erfbrsl = scp.special.erf(brsl); expbrsl = np.exp(-brsl ** 2)
    erfbrsr = scp.special.erf(brsr); expbrsr = np.exp(-brsr ** 2)

    x = lambda x, erfxsl, erfxsr, expxsl, expxsr: (expxsl - expxsr) * sqrt_sigma / sqpi + (x - bin_left) * erfxsl + (bin_right - x) * erfxsr
    x1 = x(segment_left, erfblsl, erfblsr, expblsl, expblsr)
    x2 = x(segment_right, erfbrsl, erfbrsr, expbrsl, expbrsr)

    x = lambda erfxsl, erfxsr, expxsl, expxsr: (expxsl - expxsr) * sigma / sq2pi + 0.5 * (erfxsl - erfxsr)
    x3 = x(blsl_sigma * erfblsl, blsr_sigma * erfblsr, expblsl * (segment_left + bin_left), expblsr * (segment_left + bin_right))
    x4 = x(brsl_sigma * erfbrsl, brsr_sigma * erfbrsr, expbrsl * (segment_right + bin_left), expbrsr * (segment_right + bin_right))

    C1 = 0.5 * (x2 - x1) / (bin_right - bin_left)
    C2 = 0.5 * (sigma ** 2 * (erfblsl - erfblsr - erfbrsl + erfbrsr) - x3 + x4) / (bin_right - bin_left)
    return C1, C2

def get_bin_edges(bins):
    """Convert reference wavelengths of detector bins to edge wavelengths
    
    `bins` is the array of reference wavelengths of length M. The M-1 inner bin edges are
    taken as midpoints between adjacent values of this array. The 2 outer bin edges are
    estimated by assuming that the first and the last bins are symmetric with respect to
    their reference wavelengths. The input must be strictly ascending
    
    Parameters
    ----------
    bins : array of length M
        Reference wavelengths of all detector bins
    
    Returns
    -------
    array of length M+1
        Bin edge wavelengths
    """
    if not np.all(bins[1:] > bins[:-1]):
        raise ValueError('Bin wavelengths must be strictly ascending')

    return np.concatenate([[bins[0] - (bins[1] - bins[0]) / 2], (bins[1:] + bins[:-1]) / 2, [bins[-1] + (bins[-1] - bins[-2]) / 2]])

def convolution_weights(bins, x, sigma, clip = 5.0, mode = 'window', max_size = 10e6, flush_cache = False):
    """Calculate the complete convolution matrix for an arbitrary flux density spectrum
    and an arbitrary set of detector bins
    
    The final observed spectrum is then given by `C * flux`, where `flux` is the (true) flux
    density spectrum sampled at wavelengths `x`, and `C` is the convolution matrix calculated
    by this function. Here `*` represents matrix multiplication (dot product)
    
    It is assumed that the flux density between the sampling wavelengths is given by
    linear interpolation, while beyond the range of `x` it is zero

    All of `bins`, `x` and `sigma` must be given in the same units, e.g. A

    Since the number of detector bins and flux wavelengths may be very large, the output is
    returned as a sparse matrix in the COOrdinate format. The function also checks that the
    memory usage does not exceed the limit provided in the optional argument `max_size`

    The function implements memory caching
    
    Parameters
    ----------
    bins : array of length M
        Reference wavelengths of all detector bins. See `get_bin_edges()`
    x : array of length N
        Array of wavelengths where the flux density spectrum is sampled. The segments are
        defined as the N-1 intervals between adjacent values of this array. Must be
        strictly ascending
    sigma : float or array of length M or array of length N
        Resolution of the detector, defined as the standard deviation of the Gaussian
        convolution kernel. If scalar, constant resolution is adopted for all bins and all
        segments. If `mode` is 'window', the resolution is defined at each detector bin, and
        the length of this array must be M. If `mode` is 'dispersion', the resolution is
        defined at each wavelength in the spectrum and the length of this array must be N
    clip : float, optional
        Assume that the weights for the detector bin / spectrum segment pair are zero if
        the wavelength ranges of the bin and the segment are separated by this many values of
        `sigma` for this pair. The argument removes negligibly small weights from the result,
        allowing the final sparse matrix to take up less memory
    mode : str, optional
        If 'window', the resolution determines the width of the window, over which each
        detector bin is sampling the spectrum flux density. If 'dispersion', the resolution
        determines the dispersion range, into which each segment of the spectrum
        spreads out before it is binned by the detector. There is no distinction between
        these two modes if the resolution is constant. This argument is therefore ignored
        when `sigma` is scalar
    max_size : int, optional
        Maximum number of non-zero elements in the convolution matrix. An exception is thrown
        if the predicted number of non-zero elements exceeds this argument. The number of
        non-zero elements directly correlates with memory usage
    flush_cache : bool, optional
        If True, discard cache and calculate the convolution matrix afresh
    
    Returns
    -------
    C : scipy.sparse._coo.coo_matrix
        Convolution weight matrix, such that the dot product of this matrix and the flux vector
        gives the vector of average flux densities in the detector bins
    """
    global _convolution_weights_cache
    try:
        _convolution_weights_cache
    except:
        _convolution_weights_cache = {}

    # Since input data can be very large, we will not attempt sorting it here. Leave it up to the user
    if not np.all(bins[1:] > bins[:-1]):
        raise ValueError('Bin wavelengths must be strictly ascending')
    if not np.all(x[1:] > x[:-1]):
        raise ValueError('x must be strictly ascending')

    # If `sigma` is scalar, we can choose any mode and populate the entire array with constant values
    try:
        sigma[0]
    except:
        sigma = np.full(len(bins), sigma)
        mode = 'window'

    # Check for cached output
    if not flush_cache:
        hash_string = ''.join(list(map(lambda arg: hashlib.sha256(bytes(arg)).hexdigest(), [bins, x, sigma]))) + str(clip) + mode
        if hash_string in _convolution_weights_cache:
            return _convolution_weights_cache[hash_string]

    # Check dimensions of `sigma` depending on the mode
    if mode == 'window':
        if len(bins) != len(sigma):
            raise ValueError('In "window" mode, must provide sigma for each bin')
    elif mode == 'dispersion':
        if len(x) != len(sigma):
            raise ValueError('In "dispersion" mode, must provide sigma for each x')
        # For dispersion, we want `sigma` at spectrum segments, not wavelength points
        sigma = (sigma[1:] + sigma[:-1]) / 2

    # Calculate bin edges
    bin_edges = get_bin_edges(bins)

    # Estimate the range of wavelengths that fall within the clipping range (clip * sigma) of each bin
    clip_start = np.zeros(len(bins), dtype = int)
    clip_end = np.zeros(len(bins), dtype = int)
    for i in range(len(bins)):
        if mode == 'dispersion':
            clip_indices = np.where(((x[:-1] - bin_edges[i + 1]) < (clip * sigma)) & ((bin_edges[i] - x[1:]) < (clip * sigma)))[0]
            if len(clip_indices) == 0:
                clip_start[i] = 0; clip_end[i] = 0
            else:
                clip_start[i] = np.min(clip_indices); clip_end[i] = np.max(clip_indices) + 1
        else:
            # If sigma is constant for the bin, we can find the clipping range more efficiently with np.searchsorted()
            clip_start[i] = np.searchsorted(x[1:], bin_edges[i] - clip * sigma[i])
            clip_end[i] = np.searchsorted(x[:-1], bin_edges[i + 1] + clip * sigma[i])

    # Get the row and column indices of each non-zero element of the (to be calculated) convolution matrix. Columns correspond to
    # detector bins, and rows correspond to spectrum segments
    row_lengths = clip_end - clip_start
    nonzero_count = np.sum(row_lengths)
    if nonzero_count > max_size:
        raise ValueError('Maximum memory usage exceeded. Requested: {}, allowed: {}. See the optional `max_size` argument'.format(nonzero_count, max_size))
    row_end = np.cumsum(row_lengths)
    row_start = row_end - row_lengths
    row_indices = np.zeros(nonzero_count, dtype = int)
    col_indices = np.zeros(nonzero_count, dtype = int)
    for i in range(len(bins)):
        row_indices[row_start[i] : row_end[i]] = np.arange(clip_start[i], clip_end[i])
        col_indices[row_start[i] : row_end[i]] = i

    # Obtain the values of every argument required by convolution_integral() for each non-zero element of the convolution matrix
    segment_left = x[:-1][row_indices]
    segment_right = x[1:][row_indices]
    bin_left = bin_edges[col_indices]
    bin_right = bin_edges[col_indices + 1]
    if mode == 'dispersion':
        sigma_all = sigma[row_indices]
    else:
        sigma_all = sigma[col_indices]

    # Run the convolution integral calculator
    C1, C2 = convolution_integral(sigma_all, segment_left, segment_right, bin_left, bin_right)

    # The way convolution_integral() works, we now have "observed = C1 * offset + C2 * slope". Instead, we want to convert these
    # two matrices into a single matrix C that gives "observed = C * flux". We can do this with SciPy's sparse matrix operations.
    # First, pad both C1 and C2 with empty rows and get their C1[:-1] and C1[1:] slices (low and high as they are called here)
    C1_low  = scp.sparse.coo_matrix((C1, (col_indices, row_indices + 1)), shape = [len(bins), len(x)])
    C2_low  = scp.sparse.coo_matrix((C2, (col_indices, row_indices + 1)), shape = [len(bins), len(x)])
    C1_high = scp.sparse.coo_matrix((C1, (col_indices, row_indices)), shape = [len(bins), len(x)])
    C2_high = scp.sparse.coo_matrix((C2, (col_indices, row_indices)), shape = [len(bins), len(x)])
    # Now do the same with the wavelength vector and do the conversion
    padded_x = np.insert(x, [0, len(x)], [0, 0])
    C = C1_high + (C2_low - C1_low.multiply(padded_x[:-2])).multiply(1 / (padded_x[1:-1] - padded_x[:-2])) - (C2_high - C1_high.multiply(padded_x[1:-1])).multiply(1 / (padded_x[2:] - padded_x[1:-1]))

    _convolution_weights_cache[hash_string] = C
    return C

def combine_arms(wl = None, flux = None):
    """Combine wavelengths and fluxes recorded by individual spectrograph arms into
    a single spectrum
    
    `wl` and `flux` are dictionaries, keyed by spectrograph arm identifiers (must be
    listed in `settings.arms`). The function combines the spectra in each arm into a
    single spectrum. If the wavelength ranges overlap between two arms, the overlap
    region is removed from the lower priority arm (the priorities are set in
    `settings.arms` as well)
    
    Parameters
    ----------
    wl : None or list or dict, optional
        If `wl` is a dictionary, it must be keyed by the identifiers of the arms that
        need to be combined. The values are 1D arrays of reference wavelengths of
        detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
        identifiers and the "typical" wavelength sampling for each arm (as defined in
        `settings.arms`) will be assumed. Alternatively, set to `None` to use all
        arms defined in `settings.arms` with "typical" wavelength sampling
    flux : None or dict, optional
        Dictionary of fluxes corresponding to the wavelength bins in `wl`. Must have
        the same keys and array lengths as `wl`. Alternatively, set to `None` if only
        wavelengths are required in the output
    
    Returns
    -------
    wl : array_like
        Combined and sorted array of reference wavelengths for all arms with overlapping
        wavelength bins removed according to arm priorities
    flux : array_like
        Corresponding array of fluxes. Only returned if the optional argument `flux` is
        not `None`
    """
    # Populate wl if not given
    if wl is None:
        wl = {arm: settings['arms'][arm]['wl'] for arm in settings['arms']}
    elif type(wl) is list:
        wl = {arm: settings['arms'][arm]['wl'] for arm in wl}
    else:
        wl = copy.deepcopy(wl) # We will be transforming wavelengths in place, so get a copy
    flux = copy.deepcopy(flux) # Same for flux

    # If flux is given, make sure its arms and dimensionality match wl
    if flux is not None:
        if set(wl.keys()) != set(flux.keys()):
            raise ValueError('The spectrograph arms in the `wl` and `flux` dictionaries do not match')
        if not np.all([len(wl[key]) == len(flux[key]) for key in wl]):
            raise ValueError('The dimensions of `wl` and `flux` do not match')

    # Resolve overlaps
    for arm_1, arm_2 in itertools.combinations(wl.keys(), 2):
        # We need at least two reference wavelengths to define wavelength bins
        if (len(wl[arm_1]) < 2) or (len(wl[arm_2]) < 2):
            continue
        # The overlap is evaluated for wavelength bin edges, not reference wavelengths themselves
        bin_edges_1 = get_bin_edges(wl[arm_1])
        bin_edges_2 = get_bin_edges(wl[arm_2])
        # Default priorities to zero if not set
        if 'priority' not in settings['arms'][arm_1]:
            priority_1 = 0
        else:
            priority_1 = settings['arms'][arm_1]['priority']
        if 'priority' not in settings['arms'][arm_2]:
            priority_2 = 0
        else:
            priority_2 = settings['arms'][arm_2]['priority']
        # If no overlap, do nothing
        if (np.min(bin_edges_1) > np.max(bin_edges_2)) or (np.min(bin_edges_2) > np.max(bin_edges_1)):
            continue
        # Compute the overlap region and check that priorities are different (else we don't know which arm to keep)
        overlap = [max(np.min(bin_edges_1), np.min(bin_edges_2)), min(np.max(bin_edges_1), np.max(bin_edges_2))]
        if priority_1 == priority_2:
            raise ValueError('Spectrograph arms {} and {} overlap in the region ({}:{}), but have equal priorities ({})'.format(arm_1, arm_2, *overlap, priority_2))
        if priority_1 > priority_2:
            high = bin_edges_1; low = bin_edges_2
            arm_high = arm_1; arm_low = arm_2
        else:
            high = bin_edges_2; low = bin_edges_1
            arm_high = arm_2; arm_low = arm_1
        # A higher priority arm cannot be internal to a lower priority arm, as that would slice the lower priority arm in two
        if (np.min(high) > np.min(low)) and (np.max(high) < np.max(low)):
            raise ValueError('Spectrograph arm {} (priority {}) is internal to {} (priority {}). This results in wavelength range discontinuity'.format(arm_high, max(priority_1, priority_2), arm_low, min(priority_1, priority_2)))
        # Mask out the overlap region in the lower priority arm
        mask = (low <= np.min(high)) | (low >= np.max(high))
        # Since "<=" and ">=" are used above, we may get "orphaned" bin edges, which need to be removed as well
        if (np.max(high) == np.max(low)):
            mask[low == np.max(high)] = False
        if (np.min(high) == np.min(low)):
            mask[low == np.min(high)] = False
        # Convert the bin edge mask into reference wavelength mask (both left and right edges must be defined for the bin to survive)
        mask = mask[:-1] & mask[1:]
        wl[arm_low] = wl[arm_low][mask]
        if flux is not None:
            flux[arm_low] = flux[arm_low][mask]

    # Remove empty arms
    for arm in list(wl.keys()):
        if len(wl[arm]) < 2:
            del wl[arm]
            if flux is not None:
                del flux[arm]

    # Combine the arms into a single spectrum
    keys = list(wl.keys())
    wl = np.concatenate([wl[key] for key in keys])
    if flux is not None:
        flux = np.concatenate([flux[key] for key in keys])
    sort = np.argsort(wl)
    wl = wl[sort]
    if flux is not None:
        flux = flux[sort]

    if flux is not None:
        return wl, flux
    else:
        return wl

def simulate_observation(wl, flux, detector_wl = None, mask_unmodelled = True, clip = 5, combine = True):
    """Simulate observation of the model spectrum by a spectrograph. The result is a combined
    spectrum from all arms (see `combine_arms()`)
    
    If the model does not fully cover the range of a spectrograph arm, the output flux density
    in the affected detector wavelength bins will be set to `np.nan`. This behavior can be
    disabled by setting `mask_unmodelled` to False, in which case the edge effects of the
    convolution will be left in the output spectrum
    
    Parameters
    ----------
    wl : array_like
        Model wavelengths
    flux : array_like
        Model flux densities corresponding to `wl`. It is assumed that the flux density between
        the values of `wl` can be obtained by linear interpolation, and the flux density beyond
        the range of `wl` is zero
    detector_wl : None or list or dict, optional
        Reference wavelengths of the detector bins in each arm of the spectrograph. If
        `detector_wl` is a dictionary, it must be keyed by the identifiers of the arms that
        are used in this observation. The values are 1D arrays of reference wavelengths of
        detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
        identifiers and the "typical" wavelength sampling for each arm (as defined in
        `settings.arms`) will be assumed. Alternatively, set to `None` to use all
        arms defined in `settings.arms` with "typical" wavelength sampling
    mask_unmodelled : bool, optional
        If True, set to `np.nan` the flux density in all bins that are affected by the finite
        wavelength range of the model spectrum. Otherwise, the bins near the edges of the
        model will suffer from convolution edge effects, and the bins far beyond the wavelength
        range of the model will receive zero flux
    clip : float, optional
        Sigma-clipping parameter for the convolution calculator (see `convolution_weights()`).
        This parameter also determines the range within which the lack of model coverage at a
        particular wavelength can affect the detector bins
    combine : bool, optional
        If True, return a single wavelength and a single flux array that represents the combined
        spectrum across all arms of the spectrograph (see `combine_arms()`). Otherwise, return
        the spectra in individual arms
    
    Returns
    -------
    wl : dict or array_like
        Wavelengths of the observed spectrum with the spectrograph (keyed by spectrograph arm
        if `combine` is True)
    flux : dict or array_like
        Corresponding flux densities (keyed by spectrograph arm if `combine` is True)
    """
    # Populate detector_wl if not given
    if detector_wl is None:
        detector_wl = {arm: settings['arms'][arm]['wl'] for arm in settings['arms']}
    elif type(detector_wl) is list:
        detector_wl = {arm: settings['arms'][arm]['wl'] for arm in detector_wl}

    # Resample the spectrum onto the detector bins of each arm
    detector_flux = {}
    for arm in detector_wl:
        if 'sigma' not in settings['arms'][arm]:
            settings['arms'][arm]['sigma'] = settings['arms'][arm]['FWHM'] / (2 * np.sqrt(2 * np.log(2)))
        C = convolution_weights(detector_wl[arm], wl, settings['arms'][arm]['sigma'], clip = clip)
        detector_flux[arm] = C * flux
        # Remove wavelengths that exceed the modelled range
        if mask_unmodelled:
            message = 'In spectrograph arm {} the model does not cover the full wavelength range of the detector. Affected bins were set to np.nan'.format(arm)
            first = C.getcol(0).nonzero()[0]; last = C.getcol(-1).nonzero()[0]
            if len(first) == 0 and len(last) == 0:
                # If neither the first nor the last spectrum segment contribute to detected flux, the model either
                # covers the entire range of the detector or completely misses it
                if (np.min(wl) < np.min(detector_wl[arm])) and (np.max(wl) > np.max(detector_wl[arm])):
                    continue
                else:
                    detector_flux[arm] = np.full(len(detector_flux[arm]), np.nan)
                    warn(message)
                    continue
            warn(message)
            mask = np.full(len(detector_flux[arm]), True)
            if len(first) != 0:
                mask[detector_wl[arm] <= detector_wl[arm][np.max(first)]] = False
            if len(last) != 0:
                mask[detector_wl[arm] >= detector_wl[arm][np.min(last)]] = False
            detector_flux[arm][~mask] = np.nan


    # Combine the results into a single spectrum
    if combine:
        return combine_arms(detector_wl, detector_flux)
    else:
        return detector_wl, detector_flux

class ModelGridInterpolator:
    """Handler class for interpolating the model grid to arbitrary stellar parameters
    
    The class provides the `interpolate()` method to carry out the interpolation. The
    interpolation is linear with dynamic fetching of models from disk and caching for
    loaded models
    
    Attributes
    ----------
    statistics : array_like
        Statistical data to track the interpolator's performance. Includes the following
        keys:
            'num_models_used': Total number of models read from disk *if* no caching was
                               used
            'num_models_loaded': Actual number of models read from disk
            'num_interpolations': Total number of interpolator calls
            'num_interpolators_built': Total number of interpolator objects constructed
                                       (scipy.interpolate.RegularGridInterpolator)
    """
    def __init__(self, resample = True, detector_wl = None, max_models = 1000):
        """
        Parameters
        ----------
        resample : bool, optional
            If True, resample the models to the detector bins before interpolation. The
            resampling is carried out with `simulate_observation()`. Otherwise,
            interpolate without resampling over the original wavelength grid of the models
        detector_wl : None or list or dict, optional
            Reference wavelengths of the detector bins in each arm of the spectrograph. The
            argument is only relevant if `resample` is set to True. If `detector_wl` is
            a dictionary, it must be keyed by the identifiers of the arms that
            are used in this observation. The values are 1D arrays of reference wavelengths of
            detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
            identifiers and the "typical" wavelength sampling for each arm (as defined in
            `settings.arms`) will be assumed. Alternatively, set to `None` to use all
            arms defined in `settings.arms` with "typical" wavelength sampling
        max_models : int, optional
            Maximum number of models to keep in the loaded models cache. If exceeded, the
            models loaded earliest will be removed from the cache. Higher numbers lead to
            less frequent disk access (and hence faster performance) but higher memory usage
        """
        # Populate detector_wl if not given
        if resample:
            if detector_wl is None:
                detector_wl = {arm: settings['arms'][arm]['wl'] for arm in settings['arms']}
            elif type(detector_wl) is list:
                detector_wl = {arm: settings['arms'][arm]['wl'] for arm in detector_wl}
            setattr(self, '_detector_wl', detector_wl)
        setattr(self, '_resample', resample)

        # Load grid dimensions
        setattr(self, '_grid', read_grid_dimensions())

        # Loaded models
        setattr(self, '_loaded', {})
        setattr(self, '_loaded_ordered', [])
        setattr(self, '_max_models', max_models)

        # Models embedded into the current interpolator
        setattr(self, '_interpolator_models', set())

        # Holders of statistical information
        setattr(self, 'statistics', {'num_models_used': 0, 'num_models_loaded': 0, 'num_interpolations': 0, 'num_interpolators_built': 0})

    def _build_interpolator(self, x):
        # Make sure x has the right dimensions
        if set(x.keys()) != set(self._grid.keys()):
            raise ValueError('Model grid dimensions and requested interpolation target dimensions do not match')

        # Which models are required to interpolate to x?
        subgrid = {}
        for key in self._grid.keys():
            if (x[key] > np.max(self._grid[key])) or (x[key] < np.min(self._grid[key])):
                raise ValueError('Model grid dimensions exceeded along {} axis'.format(key))
            if x[key] in self._grid[key]:
                subgrid[key] = np.array([x[key]])
            else:
                subgrid[key] = np.array([np.max(self._grid[key][self._grid[key] < x[key]]), np.min(self._grid[key][self._grid[key] > x[key]])])
        required_models = set(['|'.join(np.array(model).astype(str)) for model in itertools.product(*[subgrid[key] for key in sorted(list(subgrid.keys()))])])
        self.statistics['num_models_used'] += len(required_models)

        # If the current interpolator is already based on these models, just return it
        if required_models == self._interpolator_models:
            return self._interpolator

        # Determine which of the required models have not been loaded yet
        new_models = [model for model in required_models if model not in self._loaded]

        # If the total number of models exceeds max_models, delete the ones loaded earlier
        if len(new_models) + len(self._loaded) > self._max_models:
            to_delete = len(new_models) + len(self._loaded) - self._max_models
            remaining = []
            for model in self._loaded_ordered:
                if model in required_models or to_delete == 0:
                    remaining += [model]
                else:
                    del self._loaded[model]
                    to_delete -= 1
            self._loaded_ordered = remaining

        # Load the new models
        for model in new_models:
            params = np.array(model.split('|')).astype(float)
            keys = sorted(list(x.keys()))
            wl, flux = read_grid_model({keys[i]: params[i] for i in range(len(keys))})
            if self._resample:
                wl, flux = simulate_observation(wl, flux, self._detector_wl)
            try:
                self._wl
            except:
                setattr(self, '_wl', wl)
            self._loaded[model] = flux
            self._loaded_ordered += [model]
        self.statistics['num_models_loaded'] += len(new_models)

        # Build the interpolator
        subgrid_ordered = [subgrid[key] for key in sorted(list(subgrid.keys()))]
        meshgrid = np.meshgrid(*subgrid_ordered, indexing = 'ij')
        spectra = np.vectorize(lambda *x: self._loaded['|'.join(np.array(x).astype(str))], signature = '(),(),(),()->(n)')(*meshgrid)
        setattr(self, '_interpolator', scp.interpolate.RegularGridInterpolator(subgrid_ordered, spectra))
        self._interpolator_models = required_models
        self.statistics['num_interpolators_built'] += 1
        return self._interpolator

    def interpolate(self, params):
        """Carry out model grid interpolation
        
        Parameters
        ----------
        params : dict
            Dictionary of target stellar parameters. A value must be provided for each grid
            axis, keyed by the axis name
        
        Returns
        -------
        wl : array_like
            Grid of model wavelengths in A
        flux : array_like
            Interpolated continuum-normalized flux densities corresponding to the wavelengths
            in `wl`
        """
        self.statistics['num_interpolations'] += 1
        interpolator = self._build_interpolator(params)
        return self._wl, interpolator([params[key] for key in sorted(list(params.keys()))])[0]

    def __call__(self, params):
        return self.interpolate(params)

def ranges_to_mask(arr, ranges, in_range_value = True, strict = False):
    """Convert a list of value ranges into a boolean mask, such that all values in `arr` that
    fall in any of the ranges correspond to `in_range_value`, and the rest correspond to
    `not in_range_value`
    
    Parameters
    ----------
    arr : array_like
        Array of values
    ranges : list of tuple
        List of two-element tuples, corresponding to the lower and upper bounds of each range
    in_range_value : bool, optional
        If True, in-range elements are set to True, and the rest are set to False in the mask.
        Otherwise, in-range values are set to False, and the rest are set to True
    strict : bool, optional
        If True, use strict comparison (`lower_bound < value < upper_bound`). Otherwise, use
        non-strict comparison (`lower_bound <= value <= upper_bound`)
    
    Returns
    -------
    array_like
        Boolean mask array of the same shape as `arr`
    """
    mask = np.full(np.shape(arr), not in_range_value)
    for window in ranges:
        if strict:
            mask[(arr > window[0]) & (arr < window[1])] = in_range_value
        else:
            mask[(arr >= window[0]) & (arr <= window[1])] = in_range_value
    return mask

def estimate_continuum(wl, flux, ivar, npix = 100, k = 3, masks = None):
    """Estimate continuum level in the spectrum using a spline fit
    
    The function carries out a weighted spline fit to a spectrum given by wavelengths in `wl`,
    flux densities in `flux` and using the weights in `ivar` (usually inverted variances)

    The wavelength regions given in `settings['masks']['continuum']` are excluded from the fit.
    If one of the excluded regions overlaps with the edge of the spectrum, the spline fit near
    the edge may be poorly conditioned (the spline will be extrapolated in that region, leading
    to potentially very large edge effects). If that part of the continuum is then used in
    stellar parameter determination, extremely poor convergence is likely. As such, when the
    optional `masks` argument is given, the affected region will also be removed from the main
    fitter
    
    Parameters
    ----------
    wl : array_like
        Spectrum wavelengths
    flux : array_like
        Spectrum flux densities
    ivar : array_like
        Spectrum weights (inverted variances)
    npix : int, optional
        Desired interval between spline knots in pixels. The actual interval will be adjusted
        to keep the number of pixels in each spline segment identical
    k : int, optional
        Spline degree. Defaults to cubic
    masks : dict, optional
        Dictionary of boolean masks, keyed by stellar parameters. If given, this argument will be
        modified to exclude the regions of the spectrum potentially affected by spline extrapolation
        from the main fitter
    
    Returns
    -------
    array_like
        Estimated continuum multiplier at each wavelength in `wl`
    """
    mask = (flux >= 0.0) & (ivar > 0) & (~np.isnan(ivar)) & (~np.isnan(flux))
    for bad_continuum_range in settings['masks']['continuum']:
        bad_continuum_range_mask = ranges_to_mask(wl, [bad_continuum_range], False)
        # Check for potential edge effects and remove the affected region from the fit
        if masks is not None:
            if (not bad_continuum_range_mask[mask][-1]) or (not bad_continuum_range_mask[mask][0]):
                warn('Region {} excluded from continuum estimation overflows the spectral range. To avoid edge effects, this region will be ignored by the fitter'.format(bad_continuum_range))
                for arm in masks:
                    masks[arm] &= ranges_to_mask(wl, [bad_continuum_range], False)
        mask &= bad_continuum_range_mask

    # Fit the spline
    t = wl[mask][np.round(np.linspace(0, len(wl[mask]), int(len(wl[mask]) / npix))).astype(int)[1:-1]]
    spline = scp.interpolate.splrep(wl[mask], flux[mask], w = ivar[mask], t = t, k = k)
    return scp.interpolate.splev(wl, spline)

def fit_model(wl, flux, ivar, cont, initial, priors, dof, errors, masks, interpolator):
    """Fit the model to the spectrum
    
    Helper function to `chemfit()`. It sets up a model callback for `scp.optimize.curve_fit()` with
    the appropriate signature, defines the initial guesses and bounds for all free parameters, applies
    parameter masks and initiates the optimization routine

    The results of the fit and the associated errors are placed in the `initial` and `errors` arguments
    
    Parameters
    ----------
    wl : array_like
        Spectrum wavelengths
    flux : array_like
        Spectrum flux densities
    ivar : array_like
        Spectrum weights (inverted variances)
    cont : array_like
        Continuum multiplier. The model flux density is obtained by multiplying the output of the model
        interpolator and this factor
    initial : dict
        Initial guesses for the stellar parameters, keyed by parameter. The updated parameters after the
        optimization are stored in this dictionary as well
    priors : dict of 2-element tuples
        Prior estimates of the stellar parameters, keyed by parameter. Each element is a tuple with the
        first element storing the best estimate and the second element storing its uncertainty. All tuples
        of length other than 2 are ignored
    dof : list
        List of parameters to be optimized. The rest are treated as fixed to their initial values
    errors : dict
        Estimate errors in the best-fit parameter values are placed in this dictionary
    masks : dict
        Dictionary of boolean masks, keyed by stellar parameters. The masks determine which wavelengths are
        included in the fit for each parameter. If multiple parameters are fit simultaneously, the masks are
        logically added (or)
    interpolator : ModelGridInterpolator
        Model grid interpolator object that will be used to construct models during optimization
    Returns
    -------
    array_like
        Covariance matrix of the fit
    """

    # This function will be passed to curve_fit() as the model callback. The <signature> comment is a placeholder
    # to be replaced with the interpretation of the function signature later
    def f(data_wl, params, mask, cont = cont, priors = priors, interpolator = interpolator):
        # <signature>

        # Load the requested model
        model_wl, model_flux = interpolator(params)
        model_wl = model_wl[mask]; model_flux = (cont * model_flux)[mask]

        # Add priors
        index = 1
        for param in sorted(list(params.keys())):
            if len(np.atleast_1d(priors[param])) == 2:
                model_wl = np.concatenate([np.array([-index]), model_wl])
                model_flux = np.concatenate([np.array([params[param]]), model_flux])
                index += 1

        return model_flux

    # Define p0 and bounds. The bounds are set to the model grid range stored in the interpolator object
    p0 = [initial[param] for param in dof]
    bounds = np.array([[np.min(interpolator._grid[axis]), np.max(interpolator._grid[axis])] for axis in dof]).T

    # Construct  and apply the fitting mask by superimposing the masks of individual parameters and removing bad pixels
    mask = np.full(len(wl), False)
    for param in dof:
        mask |= masks[param]
    mask &= (flux >= 0.0) & (ivar > 0) & (~np.isnan(ivar)) & (~np.isnan(flux))
    mask &= ~np.isnan(interpolator(initial))[1]
    wl = wl[mask]; flux = flux[mask]; ivar = ivar[mask]

    # Since we do not a priori know the number of parameters being fit, we need to dynamically update the signature of the
    # model callback, f(). Unfortunately, there appears to be no better way to do that than by retrieving the source code
    # of the function with inspect, updating it, and reevaluating with exec()
    f = inspect.getsource(f).split('\n')
    f[0] = f[0].replace('params', ', '.join(dof))
    f[0] = f[0].replace('mask', 'mask = mask')
    f[1] = f[1].replace('# <signature>', 'params = {' + ', '.join(['\'{}\': {}'.format(param, [param, initial[param]][param not in dof]) for param in initial]) + '}')
    scope = {'priors': priors, 'interpolator': interpolator, 'mask': mask, 'np': np, 'cont': cont}
    exec('\n'.join(f)[f[0].find('def'):], scope)
    f = scope['f']

    # Add priors. Each prior is just an extra pixel in the spectrum
    index = 1
    for param in sorted(list(priors.keys())):
        if len(np.atleast_1d(priors[param])) == 2:
            wl = np.concatenate([np.array([-index]), wl])
            flux = np.concatenate([np.array([priors[param][0]]), flux])
            ivar = np.concatenate([np.array([priors[param][1] ** -2.0]), ivar])
            index += 1

    # Run the optimizer and save the results in "initial" and "errors"
    fit = scp.optimize.curve_fit(f, wl, flux, p0 = p0, bounds = bounds, sigma = ivar ** -0.5, **settings['curve_fit'])
    for i, param in enumerate(dof):
        initial[param] = fit[0][i]
        errors[param] = np.sqrt(fit[1][i,i])
    return fit[1]

def chemfit(wl, flux, ivar, initial):
    """Determine the stellar parameters of a star given its spectrum
    
    Parameters
    ----------
    wl : dict
        Spectrum wavelengths keyed by spectrograph arm
    flux : dict
        Spectrum flux densities keyed by spectrograph arm
    ivar : dict
        Spectrum weights (inverted variances) keyed by spectrograph arm
    initial : dict
        Initial guesses for the stellar parameters, keyed by parameter. Each parameter supported
        by the model grid must be listed. The value of each element is either a float or a
        2-element tuple. In the former case, the value is treated as the initial guess to the
        fitter. Otherwise, the first element is treated as an initial guess and the second value
        is treated as the prior uncertainty in the parameter
    
    Returns
    -------
    dict
        Fitting results. Dictionary with the following keys:
            'fit': Final best-fit stellar parameters
            'errors': Standard errors in the best-fit parameters from the diagonal of covariance matrix
            'cov': Covariance matrix from the last optimizer call. This would be based on the final
                   redetermination fit if enabled, or the last step of the continuum iteration
            'cont': Best-fit continuum multiplier for the combined wavelength array (see `combine_arms()`)
            'niter': Total number of continuum iterations carried out
            'interpolator_statistics': Interpolator statistics (see `ModelGridInterpolator().statistics`)
            'warnings': Warnings issued during the fitting process
    """
    # Combine arms
    wl_combined, flux_combined = combine_arms(wl, flux)
    ivar_combined = combine_arms(wl, ivar)[1]

    # Get the fitting masks for each parameter
    masks = {}
    for param in initial:
        mask = {}
        ranges_specific = []
        ranges_general = []
        for arm in wl:
            if (arm in settings['masks']) and (param in settings['masks'][arm]):
                ranges_specific += list(settings['masks'][arm][param])
            if ('all' in settings['masks']) and (param in settings['masks']['all']):
                ranges_specific += list(settings['masks']['all'][param])
            if (arm in settings['masks']) and ('all' in settings['masks'][arm]):
                ranges_general += list(settings['masks'][arm]['all'])
            if ('all' in settings['masks']) and ('all' in settings['masks']['all']):
                ranges_general += list(settings['masks']['all']['all'])
        masks[param] = ranges_to_mask(wl_combined, ranges_specific) & ranges_to_mask(wl_combined, ranges_general)

    # Preliminary setup
    interpolator = ModelGridInterpolator(detector_wl = wl)                     # Build the interpolator and resampler
    priors = copy.deepcopy(initial)                                            # We will update "initial" to have the initial parameters at each continuum iteration.
                                                                               # Make a copy of the true initial parameters
    initial = {param: np.atleast_1d(initial[param])[0] for param in priors}    # Initial parameters do not need uncertainties
    errors = {}                                                                # Placeholder for fitting errors
    warnings_stack_length = len(warnings_stack)                                # Remember the length of pre-existing warnings stack

    # Begin continuum iterations
    niter = 0
    while True:

        fit = copy.deepcopy(initial)

        # Get a continuum estimate as spline fit to observed/model, i.e. our definition of continuum is the spline-smoothened fitting residual
        model_wl, model_flux = interpolator(fit)
        cont = estimate_continuum(wl_combined, flux_combined / model_flux, ivar_combined * model_flux ** 2, npix = settings['cont_pix_refined'], k = settings['spline_order'], masks = (None, masks)[niter == 0])

        # Run the main fitter
        for dof in settings['continuum']:
            cov = fit_model(wl_combined, flux_combined, ivar_combined, cont, fit, priors, np.atleast_1d(dof), errors, masks, interpolator)

        # Evaluate convergence
        niter += 1
        converged = [np.abs(initial[param] - fit[param]) < settings['thresholds'][param] for param in settings['thresholds']]
        if np.all(converged):
            break
        else:
            initial = copy.deepcopy(fit)
        if niter >= settings['cont_maxiter']:
            warn('Maximum number of continuum iterations reached without satisfying the convergence target')
            break

    # Run the redetermination cycle
    for dof in settings['redeterminations']:
        cov = fit_model(wl_combined, flux_combined, ivar_combined, cont, fit, priors, np.atleast_1d(dof), errors, masks, interpolator)

    # Get the texts of unique issued warnings
    warnings = np.unique(warnings_stack[warnings_stack_length:])
    inv_warnings_messages = {warnings_messages[key]: key for key in warnings_messages}
    warnings = [inv_warnings_messages[warning_id] for warning_id in warnings]

    return {'fit': fit, 'errors': errors, 'cov': cov, 'cont': cont, 'niter': niter, 'interpolator_statistics': interpolator.statistics, 'warnings': warnings}
