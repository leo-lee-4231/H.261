#include <iostream>
#include <fstream>
#include <bitset>
#include <string>
#include <map>
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

map<string, string> encode_dict;

void InitEncodeDict();

void zigzagStep(int &x, int &y, bool &flag);

bool checkCoeff(const Mat& src);

void frameEncode(int num, Mat& cache_img, bool frame_type);

string motionCompensation(Mat& y, Mat& cr, Mat& cb, const Mat cache_img, const int mb_row, const int mb_col);

void findMinimumMAD(const Mat y, const Mat cache_img, int& target_x, int& target_y, const int offset);

void fixedEncodeBlock(const Mat& src, ofstream& ofs);

void variableLengthEncodeBlock(const Mat& src, ofstream& ofs);

int main(void) {
    // init vlc encode dict
    InitEncodeDict();

    // init cache frame
    Mat cache_img;

    // encode image sequence
    for (int i = 1; i <= 113; i++) {
        // every 4 frame, the first one is I frame
        if (i % 4 == 1)
            frameEncode(i, cache_img, INTRA);
        else
            frameEncode(i, cache_img, INTER);
    }
    
    return 0;
};

void frameEncode(int num, Mat& cache_img, bool frame_type) {
    cout << "***** Encoding picture " << num << ". *****" << endl;

    /*** load image ***/
    // construct file name
    stringstream ss;
    ss << num;
    string number = ss.str();
    while (number.length() < 4)
        number.insert(0, 1, '0');
    string read_filename = "img/" + number + ".jpg";

    // read picture
    Mat img = imread(read_filename);
    const int img_cols = img.cols;
    const int img_rows = img.rows;

    // convert color space
    Mat YcrcbImg;
    cvtColor(img, YcrcbImg, COLOR_RGB2YCrCb);

    // init temporary frame cache
    // because we can not modify cache frame when doing motion prediction
    Mat temp_cache_img = Mat::zeros(img.size(), img.type());
    cvtColor(temp_cache_img, temp_cache_img, CV_RGB2YCrCb);

    // init encode file
    string output_filename = "code/" + number + ".txt";
    ofstream ofs;
    ofs.open(output_filename.c_str(), ofstream::out);

    // encode picture infomation
    string PN = bitset<8>(num).to_string();
    string PL = bitset<10>(img_cols).to_string();
    string PW = bitset<10>(img_rows).to_string();
    ofs << PN << endl << PL << endl << PW << endl;

    // calculate the macroblock infomation
    const int mb_cols = img_cols / 16;
    const int mb_rows = img_rows / 16;
    int mb_count = 0;
    
    /*** encode every macro block ***/
    for (int i = 0; i < mb_rows; i++) {
        for (int j = 0; j < mb_cols; j++) {
            // encode macroblock header
            string MN = bitset<12>(mb_count).to_string();
            string MTYPE = frame_type == INTRA ? "01" : "10";
            string MQUANT = bitset<5>(16).to_string();
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

            // motion prediction
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
                    quant_y.at<float_t>(k, l) = round(dct_y.at<float_t>(k, l) / 16);
                }
            }
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    quant_cr.at<float_t>(k, l) = round(dct_cr.at<float_t>(k, l) / 16);
                    quant_cb.at<float_t>(k, l) = round(dct_cb.at<float_t>(k, l) / 16);
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
            if (quant_y_1_flag) {
                if (frame_type) fixedEncodeBlock(quant_y_1, ofs);
                else variableLengthEncodeBlock(quant_y_1, ofs);
            }
            if (quant_y_2_flag) {
                if (frame_type) fixedEncodeBlock(quant_y_2, ofs);
                else variableLengthEncodeBlock(quant_y_2, ofs);
            }
            if (quant_y_3_flag) {
                if (frame_type) fixedEncodeBlock(quant_y_3, ofs);
                else variableLengthEncodeBlock(quant_y_3, ofs);
            }
            if (quant_y_4_flag) {
                if (frame_type) fixedEncodeBlock(quant_y_4, ofs);
                else variableLengthEncodeBlock(quant_y_4, ofs);
            }
            if (quant_cb_flag) {
                if (frame_type) fixedEncodeBlock(quant_cb, ofs);
                else variableLengthEncodeBlock(quant_cb, ofs);
            }
            if (quant_cr_flag) {
                if (frame_type) fixedEncodeBlock(quant_cr, ofs);
                else variableLengthEncodeBlock(quant_cr, ofs);
            }

            /******************************
                resconstruct for cache 
            *******************************/
            // inverse quantization
            Mat i_dct_y, i_dct_cr, i_dct_cb;
            i_dct_y = quant_y * 16;
            i_dct_cr = quant_cr * 16;
            i_dct_cb = quant_cb * 16;

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
                i_y = ref_y + i_y;
                i_cr = ref_cr + i_cr;
                i_cb = ref_cb + i_cb;
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

    // close output file
    ofs.close();

    // save the new reconstruct image to cache frame in YCrCb
    temp_cache_img.copyTo(cache_img);
}

string motionCompensation(Mat& y, Mat& cr, Mat& cb, const Mat cache_img, const int mb_row, const int mb_col) {
    /*** find motion vector ***/
    // init useful variable
    int center_x = (mb_row * 16 + 16) / 2;
    int center_y = (mb_col * 16 + 16) / 2;
    int temp_x = center_x, temp_y = center_y;
    int offset = 15 / 2;
    bool last = false;

    // 2-D logarithmic search
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

    // cal difference
    y = y - ref_y;
    cr = cr - ref_cr;
    cb = cb - ref_cb;
    
    return mv;
}

void findMinimumMAD(const Mat y, const Mat cache_img, int& target_x, int& target_y, const int offset) {
    // init useful variable
    double min_mad = DBL_MAX;
    int min_x = target_x, min_y = target_y;
    int img_rows = cache_img.rows;
    int img_cols = cache_img.cols;

    // cal 9 direction MAD
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

        // cal the MAD and compare
        Mat abs_diff = abs(y - ref_y);
        double mad = sum(abs_diff)[0] / (16 * 16);
        if (mad < min_mad) {
            min_mad = mad;
            min_x = ref_pos_x;
            min_y = ref_pos_y;
        }
    }

    // save the min MAD position for the next search
    target_x = min_x;
    target_y = min_y;
}

void fixedEncodeBlock(const Mat& src, ofstream& ofs) {
    // init useful variable
    int x = 0, y = 0;
    int run = 0;
    bool flag = ASCEND;

    // search all 64 value in zigzag way
    int t = 64;
    while (t--) {
        // deal with run value
        int value = src.at<float_t>(x, y);
        if (value != 0 || (x == 0 && y == 0)) {
            string s_run = bitset<6>(run).to_string();
            string s_value = bitset<8>(value).to_string();
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

void variableLengthEncodeBlock(const Mat& src, ofstream& ofs) {
    // init useful variable
    int x = 0, y = 0;
    int run = 0;
    bool flag = ASCEND;

    // search 64 value in zigzag way
    int t = 64;
    while (t--) {
        // deal with run value
        int value = src.at<float_t>(x, y);
        if (value != 0) {
            string s_run = bitset<6>(run).to_string();
            string s_value = bitset<8>(value).to_string();

            // check if needed using vlc encode
            string run_value = s_run + "_" + s_value;
            if (encode_dict.find(run_value) != encode_dict.end())
                ofs << encode_dict[run_value];
            else
                ofs << encode_dict["ESCAPE"] << s_run << s_value;
            run = 0;
        }
        else
            run++;
        
        // update index in zigzag way
        zigzagStep(x, y, flag);
    }
    ofs << endl;
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

bool checkCoeff(const Mat& src) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (src.at<float_t>(i, j) != 0)
                return true;
        }
    }
    return false;
}

void InitEncodeDict() {
    encode_dict["000000_00000001"] = "110";
    encode_dict["000000_11111111"] = "111";
    encode_dict["000000_00000010"] = "01000";
    encode_dict["000000_11111110"] = "01001";
    encode_dict["000000_00000011"] = "001010";
    encode_dict["000000_11111101"] = "001011";
    encode_dict["000000_00000100"] = "00001100";
    encode_dict["000000_11111100"] = "00001101";
    encode_dict["000000_00000101"] = "001001100";
    encode_dict["000000_11111011"] = "001001101";
    encode_dict["000000_00000110"] = "001000010";
    encode_dict["000000_11111010"] = "001000011";
    encode_dict["000000_00000111"] = "00000010100";
    encode_dict["000000_11111001"] = "00000010101";
    encode_dict["000000_00001000"] = "0000000111010";
    encode_dict["000000_11111000"] = "0000000111011";
    encode_dict["000000_00001001"] = "0000000110000";
    encode_dict["000000_11110111"] = "0000000110001";
    encode_dict["000000_00001010"] = "0000000100110";
    encode_dict["000000_11110110"] = "0000000100111";
    encode_dict["000000_00001011"] = "0000000100000";
    encode_dict["000000_11110101"] = "0000000100001";
    encode_dict["000000_00001100"] = "00000000110100";
    encode_dict["000000_11110100"] = "00000000110101";
    encode_dict["000000_00001101"] = "00000000110010";
    encode_dict["000000_11110011"] = "00000000110011";
    encode_dict["000000_00001110"] = "00000000110000";
    encode_dict["000000_11110010"] = "00000000110001";
    encode_dict["000000_00001111"] = "00000000101110";
    encode_dict["000000_11110001"] = "00000000101111";
    
    encode_dict["000001_00000001"] = "0110";
    encode_dict["000001_11111111"] = "0111";
    encode_dict["000001_00000010"] = "0001100";
    encode_dict["000001_11111110"] = "0001101";
    encode_dict["000001_00000011"] = "001001010";
    encode_dict["000001_11111101"] = "001001011";
    encode_dict["000001_00000100"] = "00000011000";
    encode_dict["000001_11111100"] = "00000011001";
    encode_dict["000001_00000101"] = "0000000110110";
    encode_dict["000001_11111011"] = "0000000110111";
    encode_dict["000001_00000110"] = "00000000101100";
    encode_dict["000001_11111010"] = "00000000101101";
    encode_dict["000001_00000111"] = "00000000101010";
    encode_dict["000001_11111001"] = "00000000101011";
    
    encode_dict["000010_00000001"] = "01010";
    encode_dict["000010_11111111"] = "01011";
    encode_dict["000010_00000010"] = "00001000";
    encode_dict["000010_11111110"] = "00001001";
    encode_dict["000010_00000011"] = "00000010110";
    encode_dict["000010_11111101"] = "00000010111";
    encode_dict["000010_00000100"] = "0000000101000";
    encode_dict["000010_11111100"] = "0000000101001";
    encode_dict["000010_00000101"] = "00000000101000";
    encode_dict["000010_11111011"] = "00000000101001";

    encode_dict["000011_00000001"] = "001110";
    encode_dict["000011_11111111"] = "001111";
    encode_dict["000011_00000010"] = "001001000";
    encode_dict["000011_11111110"] = "001001001";
    encode_dict["000011_00000011"] = "0000000111000";
    encode_dict["000011_11111101"] = "0000000111001";
    encode_dict["000011_00000100"] = "00000000100110";
    encode_dict["000011_11111100"] = "00000000100111";

    encode_dict["000100_00000001"] = "001100";
    encode_dict["000100_11111111"] = "001101";
    encode_dict["000100_00000010"] = "00000011110";
    encode_dict["000100_11111110"] = "00000011111";
    encode_dict["000100_00000011"] = "0000000100100";
    encode_dict["000100_11111101"] = "0000000100101";

    encode_dict["000101_00000001"] = "0001110";
    encode_dict["000101_11111111"] = "0001111";
    encode_dict["000101_00000010"] = "00000010010";
    encode_dict["000101_11111110"] = "00000010011";
    encode_dict["000101_00000011"] = "00000000100100";
    encode_dict["000101_11111101"] = "00000000100101";

    encode_dict["000110_00000001"] = "0001010";
    encode_dict["000110_11111111"] = "0001011";
    encode_dict["000110_00000010"] = "0000000111100";
    encode_dict["000110_11111110"] = "0000000111101";

    encode_dict["000111_00000001"] = "0001000";
    encode_dict["000111_11111111"] = "0001001";
    encode_dict["000111_00000010"] = "0000000101010";
    encode_dict["000111_11111110"] = "0000000101011";

    encode_dict["001000_00000001"] = "00001110";
    encode_dict["001000_11111111"] = "00001111";
    encode_dict["001000_00000010"] = "0000000100010";
    encode_dict["001000_11111110"] = "0000000100011";

    encode_dict["001001_00000001"] = "00001010";
    encode_dict["001001_11111111"] = "00001011";
    encode_dict["001001_00000010"] = "00000000100010";
    encode_dict["001001_11111110"] = "00000000100011";

    encode_dict["001010_00000001"] = "001001110";
    encode_dict["001010_11111111"] = "001001111";
    encode_dict["001010_00000010"] = "00000000100000";
    encode_dict["001010_11111110"] = "00000000100001";

    encode_dict["001011_00000001"] = "001000110";
    encode_dict["001011_11111111"] = "001000111";

    encode_dict["001100_00000001"] = "001000100";
    encode_dict["001100_11111111"] = "001000101";

    encode_dict["001101_00000001"] = "001000000";
    encode_dict["001101_11111111"] = "001000001";

    encode_dict["001110_00000001"] = "00000011100";
    encode_dict["001110_11111111"] = "00000011101";

    encode_dict["001111_00000001"] = "00000011010";
    encode_dict["001111_11111111"] = "00000011011";

    encode_dict["010000_00000001"] = "00000010000";
    encode_dict["010000_11111111"] = "00000010001";

    encode_dict["010001_00000001"] = "0000000111110";
    encode_dict["010001_11111111"] = "0000000111111";

    encode_dict["010010_00000001"] = "0000000110100";
    encode_dict["010010_11111111"] = "0000000110101";

    encode_dict["010011_00000001"] = "0000000110010";
    encode_dict["010011_11111111"] = "0000000110011";

    encode_dict["010100_00000001"] = "0000000101110";
    encode_dict["010100_11111111"] = "0000000101111";

    encode_dict["010101_00000001"] = "0000000101100";
    encode_dict["010101_11111111"] = "0000000101101";

    encode_dict["010110_00000001"] = "00000000111110";
    encode_dict["010110_11111111"] = "00000000111111";

    encode_dict["010111_00000001"] = "00000000111100";
    encode_dict["010111_11111111"] = "00000000111101";

    encode_dict["011000_00000001"] = "00000000111010";
    encode_dict["011000_11111111"] = "00000000111011";

    encode_dict["011001_00000001"] = "00000000111000";
    encode_dict["011001_11111111"] = "00000000111001";

    encode_dict["011010_00000001"] = "00000000110110";
    encode_dict["011010_11111111"] = "00000000110111";

    encode_dict["ESCAPE"] = "000001";
}
