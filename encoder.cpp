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

void fixedEncodeBlock(const Mat& src, ofstream& ofs) {
    int x = 0, y = 0;
    int run = 0;
    bool flag = ASCEND;
    while (x > 8 && y > 8) {
        // deal with value
        int value = src.at<float_t>(x, y);
        if (value != 0) {
            string s_run = bitset<6>(run).to_string();
            string s_value = bitset<8>(value).to_string();
            ofs << s_run << s_value;
            run = 0;
        }
        else
            run++;
        
        // update index in zigzag way
        if (flag == ASCEND) {
            if (x == 0)
                flag = DESCEND;
            else
                x--;
            y++;
        }
        else {
            if (y == 0)
                flag = ASCEND;
            else
                y--;
            x++;
        }
    }
}

int main(void) {
    // load image
    Mat img = imread("img/0001.jpg");
    const int img_cols = img.cols;
    const int img_rows = img.rows;
    cout << "length: " << img_cols << endl;
    cout << "width: " << img_rows << endl;

    // create encode file
    ofstream ofs;
    ofs.open("code/0001.txt", ofstream::out);

    // insert picture infomation
    string PN = "00000001";
    ofs << PN;

    // convert color space
    Mat YcrcbImg = Mat::zeros(Size(img_rows, img_cols), img.type());
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
            string MN = bitset<8>(mb_count).to_string();
            string MTYPE = "01";
            string MQUANT = "01000";
            ofs << MN << MTYPE << MQUANT;

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
            cout << "extract y, cr, cb success." << endl;

            // motion detect
            string MV = "00000";
            ofs << MV;

            // do DCT transform
            Mat dct_y, dct_cr, dct_cb;
            dct(y, dct_y);
            dct(cr, dct_cr);
            dct(cb, dct_cb);
            cout << "dct transform success." << endl;

            // quantization
            Mat quant_y(16, 16, CV_32F);
            Mat quant_cr(16, 16, CV_32F);
            Mat quant_cb(16, 16, CV_32F);
            quant_y = dct_y / 8;
            quant_cr = dct_cr / 8;
            quant_cb = dct_cb / 8;

            // split quant_y
            Mat quant_y_1(quant_y, Rect(0, 0, 8, 8));
            Mat quant_y_2(quant_y, Rect(0, 8, 8, 8));
            Mat quant_y_3(quant_y, Rect(8, 0, 8, 8));
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
            ofs << CBP;
            cout << "check CBP success." << endl;

            // vlc encode coefficient
            fixedEncodeBlock(quant_y_1, ofs);
            fixedEncodeBlock(quant_y_2, ofs);
            fixedEncodeBlock(quant_y_3, ofs);
            fixedEncodeBlock(quant_y_4, ofs);
            fixedEncodeBlock(quant_cb, ofs);
            fixedEncodeBlock(quant_cr, ofs);

            mb_count++;
        }
    }

    ofs.close();

    // namedWindow("image", WINDOW_AUTOSIZE);
    // imshow("image", mb);
    // waitKey(0);
    
    return 0;
};
