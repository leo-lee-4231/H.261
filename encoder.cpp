#include <iostream>
#include <fstream>
#include <bitset>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ASCEND true
#define DESCEND false
#define INTRA true
#define INTER false

int direction[9][2] = {
    {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, 0}
};

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
}

void findMinimumMAD(const Mat y, const Mat cache_img, int& target_x, int& target_y, const int offset) {
    double min_mad = DBL_MAX;
    int min_x = target_x, min_y = target_y;
    int img_rows = cache_img.rows;
    int img_cols = cache_img.cols;
    for (int i = 0; i < 9; i++) {
        // get the next direction
        int ref_pos_x = target_x + direction[i][0] * offset;
        int ref_pos_y = target_y + direction[i][1] * offset;

        // check if the ref center is out of range
        if (ref_pos_x < 8 || ref_pos_x > img_rows - 8 || ref_pos_y < 8 || ref_pos_y > img_cols - 8) continue;

        // get the sub ref block and split the color channel
        Mat ref(cache_img, Rect(ref_pos_y - 8, ref_pos_x - 8, 16, 16));
        Mat ref_y_int, ref_y;
        extractChannel(ref, ref_y_int, 0);
        ref_y_int.convertTo(ref_y, CV_32F);

        // cal the mad and compare
        Mat abs_diff = abs(y - ref_y);

        double mad = sum(abs_diff)[0] / (16 * 16);
        if (mad < min_mad) {
            min_mad = mad;
            min_x = ref_pos_x;
            min_y = ref_pos_y;
        }
    }
    cout << "min mad: " << min_mad << endl;
    target_x = min_x;
    target_y = min_y;
}

string motionCompensation(Mat& y, Mat& cr, Mat& cb, const Mat cache_img, const int mb_row, const int mb_col) {
    // find motion vector
    int center_x = (mb_row * 16 + 16) / 2;
    int center_y = (mb_col * 16 + 16) / 2;
    int temp_x = center_x, temp_y = center_y;
    int offset = 15 / 2;
    bool last = false;
    while (!last) {
        findMinimumMAD(y, cache_img, temp_x, temp_y, offset);
        if (offset == 1) last = true;
        offset /= 2;
    }
    /*** low efficiency sequence way ***/
    // int img_rows = cache_img.rows;
    // int img_cols = cache_img.cols;
    // int min_x = center_x, min_y = center_y;
    // int temp_x, temp_y;
    // double min_mad = DBL_MAX;
    // for (int i = -15; i < 16; i++) {
    //     for (int j = -15; j < 16; j++) {
    //         temp_x = center_x + i;
    //         temp_y = center_y + j;
    //         // check if the ref center is out of range
    //         if (temp_x < 8 || temp_x > img_rows - 8 || temp_y < 8 || temp_y > img_cols - 8) continue;
    //         // get the sub ref block and split the color channel
    //         Mat ref(cache_img, Rect(temp_y - 8, temp_x - 8, 16, 16));
    //         Mat ref_y_int, ref_y;
    //         extractChannel(ref, ref_y_int, 0);
    //         ref_y_int.convertTo(ref_y, CV_32F);

    //         // cal the mad and compare
    //         Mat abs_diff = abs(y - ref_y);
    //         double mad = 0;
    //         for (int r = 0; r < 16; r++) {
    //             for (int c = 0; c < 16; c++) {
    //                 mad += abs_diff.at<float_t>(r, c);
    //             }
    //         }
    //         mad /= 16 * 16;
    //         if (mad < min_mad) {
    //             min_mad = mad;
    //             min_x = temp_x;
    //             min_y = temp_y;
    //         }
    //     }
    // }
    // cout << "min mad: " << min_mad << endl;
    // temp_x = min_x;
    // temp_y = min_y;

    // get motion vector
    string mv = bitset<5>(temp_y - center_y).to_string() + bitset<5>(temp_x - center_x).to_string();

    // get three channels of the cache block
    const Mat ref(cache_img, Rect(temp_y - 8, temp_x - 8, 16, 16));
    Mat ref_y = Mat::zeros(Size(16, 16), CV_32F);
    Mat ref_cr = Mat::zeros(Size(8, 8), CV_32F);
    Mat ref_cb = Mat::zeros(Size(8, 8), CV_32F);
    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 16; col++) {
            // 4:1:1 subsampling
            ref_y.at<float_t>(row, col) = ref.at<Vec3b>(row, col)[0];
            if (row % 2 == 1 && col % 2 == 0)
                ref_cr.at<float_t>(row / 2, col / 2) = ref.at<Vec3b>(row, col)[1];
            if (row % 2 == 0 && col % 2 == 0)
                ref_cb.at<float_t>(row / 2, col / 2) = ref.at<Vec3b>(row, col)[2];
        }
    }

    // cal y difference
    y = y - ref_y;
    cr = cr - ref_cr;
    cb = cb - ref_cb;
    
    return mv;
}

void frameEncode(int num, Mat& cache_img, bool frame_type) {
    cout << "***** Encoding picture " << num << ". *****" << endl;
    // load image
    stringstream ss;
    ss << num;
    string number = ss.str();
    while (number.length() < 4)
        number.insert(0, 1, '0');
    string read_filename = "img/" + number + ".jpg";
    Mat img = imread(read_filename);
    const int img_cols = img.cols;
    const int img_rows = img.rows;
    cout << "length: " << img_cols << endl;
    cout << "width: " << img_rows << endl;

    // init temporary frame cache
    Mat temp_cache_img = Mat::zeros(img.size(), img.type());
    cvtColor(temp_cache_img, temp_cache_img, CV_RGB2YCrCb);

    // create encode file
    string output_filename = "code/" + number + ".txt";
    ofstream ofs;
    ofs.open(output_filename.c_str(), ofstream::out);

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
            string MTYPE = frame_type == INTRA ? "01" : "10";
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
            string MV = frame_type == INTRA ? "0000000000" : motionCompensation(y, cr, cb, cache_img, i, j);
            ofs << MV << endl;

            // do DCT transform
            Mat dct_y, dct_cr, dct_cb;
            dct(y, dct_y);
            dct(cr, dct_cr);
            dct(cb, dct_cb);

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

            // vlc encode coefficient
            if (quant_y_1_flag)
                if (frame_type) fixedEncodeBlock(quant_y_1, ofs);
            if (quant_y_2_flag)
                if (frame_type) fixedEncodeBlock(quant_y_2, ofs);
            if (quant_y_3_flag)
                if (frame_type) fixedEncodeBlock(quant_y_3, ofs);
            if (quant_y_4_flag)
                if (frame_type) fixedEncodeBlock(quant_y_4, ofs);
            if (quant_cb_flag)
                if (frame_type) fixedEncodeBlock(quant_cb, ofs);
            if (quant_cr_flag)
                if (frame_type) fixedEncodeBlock(quant_cr, ofs);

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

            /*** motion compensation ***/
            if (frame_type == INTER) {
                // extract motion vector
                int mvh = bitset<5>(MV.substr(0, 5)).to_ulong();
                int mvv = bitset<5>(MV.substr(5, 5)).to_ulong();
                mvh = mvh > 16 ? mvh - 32 : mvh;
                mvv = mvv > 16 ? mvv - 32 : mvv;

                // get ref center
                int ref_pos_x = (i * 16 + 16) / 2 + mvv;
                int ref_pos_y = (j * 16 + 16) / 2 + mvh;
                Mat ref_mb(cache_img, Rect(ref_pos_y - 8, ref_pos_x - 8, 16, 16));

                // extract three channel of ref mb
                Mat ref_y = Mat::zeros(Size(16, 16), CV_32F);
                Mat ref_cr = Mat::zeros(Size(8, 8), CV_32F);
                Mat ref_cb = Mat::zeros(Size(8, 8), CV_32F);
                for (int row = 0; row < 16; row++) {
                    for (int col = 0; col < 16; col++) {
                        // 4:1:1 subsampling
                        ref_y.at<float_t>(row, col) = ref_mb.at<Vec3b>(row, col)[0];
                        if (row % 2 == 1 && col % 2 == 0)
                            ref_cr.at<float_t>(row / 2, col / 2) = ref_mb.at<Vec3b>(row, col)[1];
                        if (row % 2 == 0 && col % 2 == 0)
                            ref_cb.at<float_t>(row / 2, col / 2) = ref_mb.at<Vec3b>(row, col)[2];
                    }
                }

                // do the compensation
                i_y = i_y + ref_y;
                i_cr = i_cr + ref_cr;
                i_cb = i_cb + ref_cb;
            }

            // assign to cache frame
            Mat cache_mb(temp_cache_img, Rect(j * 16, i * 16, 16, 16));
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

    temp_cache_img.copyTo(cache_img);

    // covert color space for cache frame
    // cvtColor(cache_img, cache_img, COLOR_YCrCb2RGB);
}

int main(void) {
    Mat cache_img, RGB_img;
    namedWindow("image", WINDOW_AUTOSIZE);

    frameEncode(1, cache_img, INTRA);
    cvtColor(cache_img, RGB_img, COLOR_YCrCb2RGB);
    imshow("image", RGB_img);
    waitKey(0);

    frameEncode(2, cache_img, INTER);
    cvtColor(cache_img, RGB_img, COLOR_YCrCb2RGB);
    imshow("image", RGB_img);
    waitKey(0);

    frameEncode(3, cache_img, INTER);
    cvtColor(cache_img, RGB_img, COLOR_YCrCb2RGB);
    imshow("image", RGB_img);
    waitKey(0);

    frameEncode(4, cache_img, INTER);
    cvtColor(cache_img, RGB_img, COLOR_YCrCb2RGB);
    imshow("image", RGB_img);
    waitKey(0);
    
    return 0;
};
