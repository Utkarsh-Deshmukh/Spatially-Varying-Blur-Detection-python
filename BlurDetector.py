import cv2
import numpy as np
import os
from skimage.filters.rank import entropy
from skimage.morphology import square
import copy
import time

class BlurDetector(object):
    def __init__(self):
        self.downsampling_factor = 4
        self.num_scales = 4
        self.scale_start = 3
        self.entropy_filt_kernel_sze = 7
        self.sigma_s_RF_filter = 15
        self.sigma_r_RF_filter = 0.25
        self.num_iterations_RF_filter = 3
        self.scales = self.createScalePyramid()
        self.__freqBands = []
        self.__dct_matrices = []
        self.freq_index = []

    def disp_progress(self, i, rows, old_progress):
        progress_dict = {10:'[|                  ] 10%',
                         20:'[| |                ] 20%',
                         30:'[| | |              ] 30%',
                         40:'[| | | |            ] 40%',
                         50:'[| | | | |          ] 50%',
                         60:'[| | | | | |        ] 60%',
                         70:'[| | | | | | |      ] 70%',
                         80:'[| | | | | | | |    ] 80%',
                         90:'[| | | | | | | | |  ] 90%',
                         100:'[| | | | | | | | | |] 100%'}

        i_done = i / rows * 100;
        p_done = round(i_done / 10) * 10;
        if(p_done != old_progress):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(progress_dict[p_done])
            old_progress = p_done
        return(p_done)

    def createScalePyramid(self):
        scales = []
        for i in range(self.num_scales):
            scales.append((2**(self.scale_start + i)) - 1)          # Scales would be 7, 15, 31, 63 ...
        return(scales)

    def computeImageGradientMagnitude(self, img):
        __sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)  # Find x and y gradients
        __sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)

        # Find gradient magnitude
        __magnitude = np.sqrt(__sobelx ** 2.0 + __sobely ** 2.0)
        return(__magnitude)

    def __computeFrequencyBands(self):
        for current_scale in self.scales:
            matrixInds = np.zeros((current_scale, current_scale))

            for i in range(current_scale):
                matrixInds[0 : max(0, int(((current_scale-1)/2) - i +1)), i] = 1

            for i in range(current_scale):
                if (current_scale-((current_scale-1)/2) - i) <= 0:
                    matrixInds[0:current_scale - i - 1, i] = 2
                else:
                    matrixInds[int(current_scale - ((current_scale - 1) / 2) - i - 1): int(current_scale - i - 1), i]=2;
            matrixInds[0, 0] = 3
            self.__freqBands.append(matrixInds)

    def __dctmtx(self, n):
        [mesh_cols, mesh_rows] = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, n-1, n))
        dct_matrix = np.sqrt(2/n) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2*n));
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return(dct_matrix)

    def __createDCT_Matrices(self):
        if(len(self.__dct_matrices) > 0):
            raise TypeError("dct matrices are already defined. Redefinition is not allowed.")
        for curr_scale in self.scales:
            dct_matrix = self.__dctmtx(curr_scale)
            self.__dct_matrices.append(dct_matrix)

    def __getDCTCoefficients(self, img_blk, ind):
        rows, cols = np.shape(img_blk)
        # D = self.__dctmtx(rows)
        D = self.__dct_matrices[ind]
        dct_coeff = np.matmul(np.matmul(D, img_blk), np.transpose(D))
        return(dct_coeff)

    def entropyFilt(self, img):
        return(entropy(img, square(self.entropy_filt_kernel_sze)))

    def computeScore(self, weighted_local_entropy, T_max):
        # normalize weighted T max matrix
        min_val = weighted_local_entropy.min()
        weighted_T_Max = weighted_local_entropy - min_val
        max_val = weighted_local_entropy.max()
        weighted_T_Max = weighted_local_entropy / max_val

        score = np.median(weighted_local_entropy)
        return(score)

    def TransformedDomainRecursiveFilter_Horizontal(self, I, D, sigma):
        # Feedback Coefficient (Appendix of the paper)
        a = np.exp(-np.sqrt(2) / sigma)
        F = copy.deepcopy(I)
        V = a ** D
        rows, cols = np.shape(I)

        # Left --> Right Filter
        for i in range(1, cols):
            F[:, i] = F[:, i] + np.multiply(V[:, i], (F[:, i-1] - F[:, i]))

        # Right --> Left Filter
        for i in range(cols-2, 1, -1):
            F[:, i] = F[:, i] + np.multiply(V[:, i+1], (F[:, i + 1] - F[:, i]))

        return(F)

    def RF(self, img, joint_img):
        if(len(joint_img) == 0):
            joint_img = img
        joint_img = joint_img.astype('float64')
        joint_img = joint_img / 255

        if(len(np.shape(joint_img)) == 2):
            cols, rows = np.shape(joint_img)
            channels = 1
        elif(len(np.shape(joint_img)) == 3):
            cols, rows, channels = np.shape(joint_img)
        # Estimate horizontal and vertical partial derivatives using finite differences.
        dIcdx = np.diff(joint_img, n=1, axis=1)
        dIcdy = np.diff(joint_img, n=1, axis=0)

        dIdx = np.zeros((cols, rows));
        dIdy = np.zeros((cols, rows));

        # Compute the l1 - norm distance of neighbor pixels.
        dIdx[:, 1::] = abs(dIcdx)
        dIdy[1::, :] = abs(dIcdy)

        dHdx = (1 + self.sigma_s_RF_filter / self.sigma_r_RF_filter * dIdx)
        dVdy = (1 + self.sigma_s_RF_filter / self.sigma_r_RF_filter * dIdy)

        dVdy = np.transpose(dVdy)
        N = self.num_iterations_RF_filter
        F  = copy.deepcopy(img)
        for i in range(self.num_iterations_RF_filter):
            # Compute the sigma value for this iteration (Equation 14 of our paper).
            sigma_H_i = self.sigma_s_RF_filter * np.sqrt(3) * 2 ** (N - (i + 1)) / np.sqrt(4 ** N - 1)
            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
            F = np.transpose(F)

            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
            F = np.transpose(F)

        return(F)

    def detectBlur(self, img):
        ori_rows, ori_cols = np.shape(img)
        # perform initial gausssian smoothing
        InputImageGaus = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5, sigmaY=0.5)
        __gradient_image = self.computeImageGradientMagnitude(InputImageGaus)

        total_num_layers = 1 + sum(self.scales)

        # create all dct_matrices beforehand to save computation time
        self.__createDCT_Matrices()

        # Create Frequency Labels at all the scalesv
        self.__computeFrequencyBands()

        # Compute the indices of the high frequency content inside each frequency band
        for i in range(self.num_scales):
            curr_freq_band = self.__freqBands[i]
            self.freq_index.append(np.where(curr_freq_band == 0))

        __padded_image = np.pad(__gradient_image, int(np.floor(max(self.scales)/2)), mode='constant')

        rows, cols = np.shape(__padded_image)
        L = []

        total_num_points = len([i for i in range(int(max(self.scales)/2), rows - int(max(self.scales)/2), self.downsampling_factor)]) * len([j for j in range(int(max(self.scales) / 2), cols - int(max(self.scales) / 2), self.downsampling_factor)])
        L = np.zeros((total_num_points, total_num_layers))

        iter = 0
        n = 0
        old_progress = 0
        for i in range(int(max(self.scales)/2), rows - int(max(self.scales)/2), self.downsampling_factor):
            old_progress = self.disp_progress(i, rows, old_progress)
            m = 0
            n += 1
            for j in range(int(max(self.scales) / 2), cols - int(max(self.scales) / 2), self.downsampling_factor):
                m += 1
                high_freq_components = []
                for ind, curr_scale in enumerate(self.scales):
                    Patch = __padded_image[i-np.int(curr_scale/2) : i+np.int(curr_scale/2) + 1, j-np.int(curr_scale/2) : j+np.int(curr_scale/2) + 1]
                    dct_coefficients = np.abs(self.__getDCTCoefficients(Patch, ind))

                    # store all high frequency components
                    high_freq_components.append(dct_coefficients[self.freq_index[ind]])

                # Find the first `total_num_layers` smallest values in all the high frequency components - we must not sort the entire array since that is very inefficient
                high_freq_components = np.hstack(high_freq_components)
                result = np.argpartition(high_freq_components, total_num_layers)
                L[iter, :] = high_freq_components[result[:total_num_layers]]
                iter += 1


        L = np.array(L)

        # normalize the L matrix
        for i in range(total_num_layers):
            max_val = max(L[:, i])
            L[:, i] = L[:, i] / max_val

        # perform max pooling on the normalized frequencies
        ind1d = 0
        T_max = np.zeros((n, m))
        max_val = 0
        min_val = 99999
        for i in range(n):
            for j in range(m):
                T_max[i][j] = max(L[ind1d, :])
                max_val = max(max_val, T_max[i][j])
                min_val = min(min_val, T_max[i][j])
                ind1d += 1

        # Final Map and Post Processing
        local_entropy = self.entropyFilt(T_max)
        weighted_local_entropy = np.multiply(local_entropy, T_max)

        score = self.computeScore(weighted_local_entropy, T_max)
        rows, cols = np.shape(weighted_local_entropy)

        # resize the input image to match the size of local_entropy matrix
        resized_input_image = cv2.resize(InputImageGaus, (cols, rows))
        aSmooth = cv2.GaussianBlur(resized_input_image, (3, 3), sigmaX=1, sigmaY=1)
        final_map = self.RF(weighted_local_entropy, aSmooth)

        # resize the map to the original resolution
        final_map = cv2.resize(final_map, (ori_cols, ori_rows))

        # normalize the map
        final_map = final_map / np.max(final_map)
        return(final_map)