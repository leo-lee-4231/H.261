#include <iostream>
#include <fstream>
#include <bitset>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ASCEND true
#define DESCEND false

bool checkCoeff(const Mat& src) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (src.at<float_t>(i, j) != 0)
                return true;
        }
    }
    return false;
}

void zigzagStep(int &x, int &y, bool &flag) {
    if (flag == ASCEND) {
        if (x == 0 || y == 7) {
            flag = DESCEND;
            if (y == 7) x++;
            else y++;
        }
        else {
            x--;
            y++;
        }
    }
    else {
        if (y == 0 || x == 7) {
            flag = ASCEND;
            if (x == 7) y++;
            else x++;
        }
        else {
            y--;
            x++;
        }
    }
}

void fixedEncodeBlock(const Mat& src, ofstream& ofs) {
    int x = 0, y = 0;
    int run = 0;
    bool flag = ASCEND;
    int t = 64;
    while (t--) {
        // deal with run value
        int value = src.at<float_t>(x, y);
        if (value != 0 || (x == 0 && y == 0)) {
            string s_run = bitset<6>(run).to_string();
            string s_value = bitset<9>(value).to_string();
            ofs << s_run << s_value;
            run = 0;
        }
        else
            run++;
        
        // update index in zigzag way
        zigzagStep(x, y, flag);
    }
    ofs << endl;
    cout << "end of block." << endl;
}

void iframeEncode(int num, Mat& cache_img) {
    cout << "***** Encoding picture " << num << ". *****" << endl;
    // load image
    string read_filename = "img/" + bitset<4>(num).to_string() + ".jpg";
    Mat img = imread(read_filename);
    const int img_cols = img.cols;
    const int img_rows = img.rows;
    cout << "length: " << img_cols << endl;
    cout << "width: " << img_rows << endl;

    // init frame cache
    cache_img = Mat::zeros(img.size(), img.type());

    // create encode file
    string output_filename = "code/" + bitset<4>(num).to_string() + ".txt";
    ofstream ofs;
    ofs.open("code/0001.txt", ofstream::out);

    // insert picture infomation
    string PN = bitset<8>(num).to_string();
    string PL = bitset<10>(img_cols).to_string();
    string PW = bitset<10>(img_rows).to_string();
    ofs << PN << endl << PL << endl << PW << endl;

    // convert color space
    Mat YcrcbImg;
    cvtColor(img, YcrcbImg, COLOR_RGB2YCrCb);

    // for the macroblock
    const int mb_cols = img_cols / 16;
    const int mb_rows = img_rows / 16;
    cout << "mb length: " << mb_cols << endl;
    cout << "mb width: " << mb_rows << endl;
    int mb_count = 0;
    
    for (int i = 0; i < mb_rows; i++) {
        for (int j = 0; j < mb_cols; j++) {
            cout << "dealing macroblock " << i << ", " << j << endl;
            // set macroblock header
            string MN = bitset<12>(mb_count).to_string();
            string MTYPE = "01";
            string MQUANT = "01000";
            ofs << MN << endl << MTYPE << endl << MQUANT << endl;

            // get macroblock data
            Mat mb(YcrcbImg, Rect(j * 16, i * 16, 16, 16));
            
            // extract Y, Cr, Cb block
            Mat y = Mat::zeros(Size(16, 16), CV_32F);
            Mat cr = Mat::zeros(Size(8, 8), CV_32F);
            Mat cb = Mat::zeros(Size(8, 8), CV_32F);
            for (int row = 0; row < 16; row++) {
                for (int col = 0; col < 16; col++) {
                    // 4:1:1 subsampling
                    y.at<float_t>(row, col) = mb.at<Vec3b>(row, col)[0];
                    if (row % 2 == 1 && col % 2 == 0)
                        cr.at<float_t>(row / 2, col / 2) = mb.at<Vec3b>(row, col)[1];
                    if (row % 2 == 0 && col % 2 == 0)
                        cb.at<float_t>(row / 2, col / 2) = mb.at<Vec3b>(row, col)[2];
                }
            }

            // motion detect
            string MV = "000000000000";
            ofs << MV << endl;

            // do DCT transform
            Mat dct_y, dct_cr, dct_cb;
            dct(y, dct_y);
            dct(cr, dct_cr);
            dct(cb, dct_cb);
            cout << "dct transform success." << endl;

            // quantization
            Mat quant_y = Mat::zeros(Size(16, 16), CV_32F);
            Mat quant_cr = Mat::zeros(Size(8, 8), CV_32F);
            Mat quant_cb = Mat::zeros(Size(8, 8), CV_32F);
            for (int k = 0; k < 16; k++) {
                for (int l = 0; l < 16; l++) {
                    quant_y.at<float_t>(k, l) = round(dct_y.at<float_t>(k, l) / 8);
                }
            }
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    quant_cr.at<float_t>(k, l) = round(dct_cr.at<float_t>(k, l) / 8);
                    quant_cb.at<float_t>(k, l) = round(dct_cb.at<float_t>(k, l) / 8);
                }
            }

            // split quant_y
            Mat quant_y_1(quant_y, Rect(0, 0, 8, 8));
            Mat quant_y_2(quant_y, Rect(8, 0, 8, 8));
            Mat quant_y_3(quant_y, Rect(0, 8, 8, 8));
            Mat quant_y_4(quant_y, Rect(8, 8, 8, 8));

            // check CBP
            bool quant_y_1_flag = checkCoeff(quant_y_1);
            bool quant_y_2_flag = checkCoeff(quant_y_2);
            bool quant_y_3_flag = checkCoeff(quant_y_3);
            bool quant_y_4_flag = checkCoeff(quant_y_4);
            bool quant_cb_flag = checkCoeff(quant_cb);
            bool quant_cr_flag = checkCoeff(quant_cr);
            int cbp_count = 32 * quant_y_1_flag + 16 * quant_y_2_flag + 8 * quant_y_3_flag + 
                            4 * quant_y_4_flag + 2 * quant_cb_flag + 1 * quant_cr_flag;
            string CBP = bitset<6>(cbp_count).to_string();
            ofs << CBP << endl;
            cout << "check CBP success." << endl;

            // vlc encode coefficient
            fixedEncodeBlock(quant_y_1, ofs);
            fixedEncodeBlock(quant_y_2, ofs);
            fixedEncodeBlock(quant_y_3, ofs);
            fixedEncodeBlock(quant_y_4, ofs);
            fixedEncodeBlock(quant_cb, ofs);
            fixedEncodeBlock(quant_cr, ofs);

            /******************************
                resconstruct for cache 
            *******************************/
            // inverse quantization
            Mat i_dct_y, i_dct_cr, i_dct_cb;
            i_dct_y = quant_y * 8;
            i_dct_cr = quant_cr * 8;
            i_dct_cb = quant_cb * 8;

            // inverse dct
            Mat i_y, i_cr, i_cb;
            dct(i_dct_y, i_y, cv::DCT_INVERSE);
            dct(i_dct_cr, i_cr, cv::DCT_INVERSE);
            dct(i_dct_cb, i_cb, cv::DCT_INVERSE);

            // assign to cache frame
            Mat cache_mb(cache_img, Rect(j * 16, i * 16, 16, 16));
            for (int row = 0; row < 16; row++) {
                for (int col = 0; col < 16; col++) {
                    // 4:1:1 upsampling
                    cache_mb.at<Vec3b>(row, col)[0] = i_y.at<float_t>(row, col);
                    cache_mb.at<Vec3b>(row, col)[1] = i_cr.at<float_t>(row / 2, col / 2);
                    cache_mb.at<Vec3b>(row, col)[2] = i_cb.at<float_t>(row / 2, col / 2);
                }
            }

            mb_count++;
        }
    }

    ofs.close();

    // covert color space for cache frame
    cvtColor(cache_img, cache_img, COLOR_YCrCb2RGB);
}

int main(void) {
    Mat cache_img;
    iframeEncode(1, cache_img);

    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", cache_img);
    waitKey(0);
    
    return 0;
};
