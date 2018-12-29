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

map<string, string> decode_dict;

void initDecodeDict();

void zigzagStep(int &x, int &y, bool &flag);

void frameDecode(int num, Mat& cache_img);

void decodeFixedBlock(Mat& block, ifstream& ifs);

void decodeVariableLengthBlock(Mat& block, ifstream& ifs);

int main(void) {
    // init VLC decode dict
    initDecodeDict();

    // init a cache frame
    Mat cache_img;

    // decode the image sequence
    for (int i = 1; i <= 113; i++) {
        frameDecode(i, cache_img);
    }

    return 0;
}

void frameDecode(int num, Mat& cache_img) {
    cout << "***** Decoding picture " << num << ". *****" << endl;
    
    /*** load code ***/
    // construct file name
    stringstream ss;
    ss << num;
    string number = ss.str();
    while (number.length() < 4)
        number.insert(0, 1, '0');
    string read_filename = "code/" + number + ".txt";

    // open file
    ifstream ifs;
    ifs.open(read_filename.c_str(), ifstream::in);

    // load PN, PL, PW
    string PN, PL, PW;
    ifs >> PN >> PL >> PW;
    int img_num = bitset<8>(PN).to_ulong();
    int img_cols = bitset<10>(PL).to_ulong();
    int img_rows = bitset<10>(PW).to_ulong();

    // init a Mat for reconstruct frame and convert its color space
    Mat img = Mat::zeros(Size(img_cols, img_rows), CV_8UC3);
    cvtColor(img, img, CV_RGB2YCrCb);

    /*** extract macroblock ***/
    int mb_rows = img_rows / 16;
    int mb_cols = img_cols / 16;
    int mb_count = mb_rows * mb_cols;
    while (mb_count--) {
        // load macroblock parameters
        string MN, MTYPE, MQUANT, MV, CBP;
        ifs >> MN >> MTYPE >> MQUANT >> MV >> CBP;
        int mn = bitset<12>(MN).to_ulong();
        int mtype = bitset<2>(MTYPE).to_ulong();
        int mquant = bitset<5>(MQUANT).to_ulong();
        int mvh = bitset<5>(MV.substr(0, 5)).to_ulong();
        int mvv = bitset<5>(MV.substr(5)).to_ulong();
        int cbp = bitset<6>(CBP).to_ulong();
        bool frame_type = mtype == 1 ? INTRA : INTER;

        // fix the sign of mv
        mvh = mvh > 16 ? mvh - 32 : mvh;
        mvv = mvv > 16 ? mvv - 32 : mvv;

        // create macro block coeffient container
        Mat quant_y = Mat::zeros(Size(16, 16), CV_32F);
        Mat quant_cb = Mat::zeros(Size(8, 8), CV_32F);
        Mat quant_cr = Mat::zeros(Size(8, 8), CV_32F);

        // split quant_y 
        Mat quant_y_1(quant_y, Rect(0, 0, 8, 8));
        Mat quant_y_2(quant_y, Rect(8, 0, 8, 8));
        Mat quant_y_3(quant_y, Rect(0, 8, 8, 8));
        Mat quant_y_4(quant_y, Rect(8, 8, 8, 8));

        // decode coeffient optionly by cbp value
        if (cbp & 32) {
            if (frame_type) decodeFixedBlock(quant_y_1, ifs);
            else decodeVariableLengthBlock(quant_y_1, ifs);
        }
        if (cbp & 16) {
            if (frame_type) decodeFixedBlock(quant_y_2, ifs);
            else decodeVariableLengthBlock(quant_y_2, ifs);
        }
        if (cbp & 8) {
            if (frame_type) decodeFixedBlock(quant_y_3, ifs);
            else decodeVariableLengthBlock(quant_y_3, ifs);
        }
        if (cbp & 4) {
            if (frame_type) decodeFixedBlock(quant_y_4, ifs);
            else decodeVariableLengthBlock(quant_y_4, ifs);
        }
        if (cbp & 2) {
            if (frame_type) decodeFixedBlock(quant_cb, ifs);
            else decodeVariableLengthBlock(quant_cb, ifs);
        }
        if (cbp & 1) {
            if (frame_type) decodeFixedBlock(quant_cr, ifs);
            else decodeVariableLengthBlock(quant_cr, ifs);
        }

        // inverse quantity
        Mat dct_y, dct_cr, dct_cb;
        dct_y = quant_y * mquant;
        dct_cr = quant_cr * mquant;
        dct_cb = quant_cb * mquant;

        // inverse dct
        Mat y, cr, cb;
        dct(dct_y, y, cv::DCT_INVERSE);
        dct(dct_cr, cr, cv::DCT_INVERSE);
        dct(dct_cb, cb, cv::DCT_INVERSE);

        // get the macro block position(left top)
        int row = 16 * (mn / mb_cols);
        int col = 16 * (mn % mb_cols);

        /*** motion compensation ***/
        if (frame_type == INTER) {
            // get reference frame's marco block
            int ref_pos_x = (row + 16) / 2 + mvv;
            int ref_pos_y = (col + 16) / 2 + mvh;
            Mat ref_mb(cache_img, Rect(ref_pos_y - 8, ref_pos_x - 8, 16, 16));

            // extract three channel of ref mb
            Mat ref_y = Mat::zeros(Size(16, 16), CV_32F);
            Mat ref_cr = Mat::zeros(Size(8, 8), CV_32F);
            Mat ref_cb = Mat::zeros(Size(8, 8), CV_32F);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    // 4:1:1 subsampling
                    ref_y.at<float_t>(i, j) = ref_mb.at<Vec3b>(i, j)[0];
                    if (i % 2 == 1 && j % 2 == 0)
                        ref_cr.at<float_t>(i / 2, j / 2) = ref_mb.at<Vec3b>(i, j)[1];
                    if (i % 2 == 0 && j % 2 == 0)
                        ref_cb.at<float_t>(i / 2, j / 2) = ref_mb.at<Vec3b>(i, j)[2];
                }
            }

            // do the compensation
            y = ref_y + y;
            cr = ref_cr + cr;
            cb = ref_cb + cb;
        }

        /*** reconstruct image macro block ***/
        Mat img_mb(img, Rect(col, row, 16, 16));
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                // 4:1:1 upsampling
                img_mb.at<Vec3b>(i, j)[0] = y.at<float_t>(i, j);
                img_mb.at<Vec3b>(i, j)[1] = cr.at<float_t>(i / 2, j / 2);
                img_mb.at<Vec3b>(i, j)[2] = cb.at<float_t>(i / 2, j / 2);
            }
        }
    }

    // close file
    ifs.close();

    // save img to cache img in YCrCb
    img.copyTo(cache_img);

    // write into file
    cv::cvtColor(img, img, cv::COLOR_YCrCb2RGB);
    string output_filename = "rebuild/" + number + ".jpg";
    imwrite(output_filename, img);
}

void decodeFixedBlock(Mat& block, ifstream& ifs) {
    // init useful variable
    int x = 0, y = 0;
    bool flag = ASCEND;

    // read code of the block
    string code;
    ifs >> code;

    /*** decode the run value ***/
    while (!code.empty()) {
        // get run and simulate the zigzag order step
        int run = bitset<6>(code.substr(0, 6)).to_ulong();
        code.erase(0, 6);
        while (run--) {
            zigzagStep(x, y, flag);
        }

        // get value and assign the value
        int value = bitset<9>(code.substr(0, 9)).to_ulong();
        code.erase(0, 9);
        value = value > 256 ? value - 512 : value;
        block.at<float_t>(x, y) = value;

        // move to the next step
        zigzagStep(x, y, flag);
    }
}

void decodeVariableLengthBlock(Mat& block, ifstream& ifs) {
    // init useful variable
    int x = 0, y = 0;
    bool flag = ASCEND;

    // read code of the block
    string code;
    ifs >> code;

    /*** decode the run-value pair ***/
    while (!code.empty()) {
        // decode variable length code
        string str;
        int check_length = 0;
        while (decode_dict.find(str) == decode_dict.end()) {
            check_length++;
            str = code.substr(0, check_length);
        }
        code.erase(0, check_length);
        string run_value = decode_dict[str];

        // deal with not vlc encoded run-value pair
        if (run_value == "ESCAPE") {
            run_value = code.substr(0, 6) + "_" + code.substr(6, 9);
            code.erase(0, 15);
        }

        // get run and simulate the zigzag order steps
        int run = bitset<6>(run_value.substr(0, 6)).to_ulong();
        while (run--) {
            zigzagStep(x, y, flag);
        }

        // get value and assign the value
        int value = bitset<9>(run_value.substr(7, 9)).to_ulong();
        value = value > 256 ? value - 512 : value;
        block.at<float_t>(x, y) = value;

        // move to the next step
        zigzagStep(x, y, flag);
    }
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

void initDecodeDict() {
    decode_dict["110"] =                "000000_000000001";
    decode_dict["111"] =                "000000_111111111";
    decode_dict["01000"] =              "000000_000000010";
    decode_dict["01001"] =              "000000_111111110";
    decode_dict["001010"] =             "000000_000000011";
    decode_dict["001011"] =             "000000_111111101";
    decode_dict["00001100"] =           "000000_000000100";
    decode_dict["00001101"] =           "000000_111111100";
    decode_dict["001001100"] =          "000000_000000101";
    decode_dict["001001101"] =          "000000_111111011";
    decode_dict["001000010"] =          "000000_000000110";
    decode_dict["001000011"] =          "000000_111111010";
    decode_dict["00000010100"] =        "000000_000000111";
    decode_dict["00000010101"] =        "000000_111111001";
    decode_dict["0000000111010"] =      "000000_000001000";
    decode_dict["0000000111011"] =      "000000_111111000";
    decode_dict["0000000110000"] =      "000000_000001001";
    decode_dict["0000000110001"] =      "000000_111110111";
    decode_dict["0000000100110"] =      "000000_000001010";
    decode_dict["0000000100111"] =      "000000_111110110";
    decode_dict["0000000100000"] =      "000000_000001011";
    decode_dict["0000000100001"] =      "000000_111110101";
    decode_dict["00000000110100"] =     "000000_000001100";
    decode_dict["00000000110101"] =     "000000_111110100";
    decode_dict["00000000110010"] =     "000000_000001101";
    decode_dict["00000000110011"] =     "000000_111110011";
    decode_dict["00000000110000"] =     "000000_000001110";
    decode_dict["00000000110001"] =     "000000_111110010";
    decode_dict["00000000101110"] =     "000000_000001111";
    decode_dict["00000000101111"] =     "000000_111110001";

    decode_dict["0110"] =               "000001_000000001";
    decode_dict["0111"] =               "000001_111111111";
    decode_dict["0001100"] =            "000001_000000010";
    decode_dict["0001101"] =            "000001_111111110";
    decode_dict["001001010"] =          "000001_000000011";
    decode_dict["001001011"] =          "000001_111111101";
    decode_dict["00000011000"] =        "000001_000000100";
    decode_dict["00000011001"] =        "000001_111111100";
    decode_dict["0000000110110"] =      "000001_000000101";
    decode_dict["0000000110111"] =      "000001_111111011";
    decode_dict["00000000101100"] =     "000001_000000110";
    decode_dict["00000000101101"] =     "000001_111111010";
    decode_dict["00000000101010"] =     "000001_000000111";
    decode_dict["00000000101011"] =     "000001_111111001";
    
    decode_dict["01010"] =              "000010_000000001";
    decode_dict["01011"] =              "000010_111111111";
    decode_dict["00001000"] =           "000010_000000010";
    decode_dict["00001001"] =           "000010_111111110";
    decode_dict["00000010110"] =        "000010_000000011";
    decode_dict["00000010111"] =        "000010_111111101";
    decode_dict["0000000101000"] =      "000010_000000100";
    decode_dict["0000000101001"] =      "000010_111111100";
    decode_dict["00000000101000"] =     "000010_000000101";
    decode_dict["00000000101001"] =     "000010_111111011";

    decode_dict["001110"] =             "000011_000000001";
    decode_dict["001111"] =             "000011_111111111";
    decode_dict["001001000"] =          "000011_000000010";
    decode_dict["001001001"] =          "000011_111111110";
    decode_dict["0000000111000"] =      "000011_000000011";
    decode_dict["0000000111001"] =      "000011_111111101";
    decode_dict["00000000100110"] =     "000011_000000100";
    decode_dict["00000000100111"] =     "000011_111111100";
    
    decode_dict["001100"] =             "000100_000000001";
    decode_dict["001101"] =             "000100_111111111";
    decode_dict["00000011110"] =        "000100_000000010";
    decode_dict["00000011111"] =        "000100_111111110";
    decode_dict["0000000100100"] =      "000100_000000011";
    decode_dict["0000000100101"] =      "000100_111111101";

    decode_dict["0001110"] =            "000101_000000001";
    decode_dict["0001111"] =            "000101_111111111";
    decode_dict["00000010010"] =        "000101_000000010";
    decode_dict["00000010011"] =        "000101_111111110";
    decode_dict["00000000100100"] =     "000101_000000011";
    decode_dict["00000000100101"] =     "000101_111111101";

    decode_dict["0001010"] =            "000110_000000001";
    decode_dict["0001011"] =            "000110_111111111";
    decode_dict["0000000111100"] =      "000110_000000010";
    decode_dict["0000000111101"] =      "000110_111111110";

    decode_dict["0001000"] =            "000111_000000001";
    decode_dict["0001001"] =            "000111_111111111";
    decode_dict["0000000101010"] =      "000111_000000010";
    decode_dict["0000000101011"] =      "000111_111111110";

    decode_dict["00001110"] =           "001000_000000001";
    decode_dict["00001111"] =           "001000_111111111";
    decode_dict["0000000100010"] =      "001000_000000010";
    decode_dict["0000000100011"] =      "001000_111111110";

    decode_dict["00001010"] =           "001001_000000001";
    decode_dict["00001011"] =           "001001_111111111";
    decode_dict["00000000100010"] =     "001001_000000010";
    decode_dict["00000000100011"] =     "001001_111111110";

    decode_dict["001001110"] =          "001010_000000001";
    decode_dict["001001111"] =          "001010_111111111";
    decode_dict["00000000100000"] =     "001010_000000010";
    decode_dict["00000000100001"] =     "001010_111111110";

    decode_dict["001000110"] =          "001011_000000001";
    decode_dict["001000111"] =          "001011_111111111";

    decode_dict["001000100"] =          "001100_000000001";
    decode_dict["001000101"] =          "001100_111111111";

    decode_dict["001000000"] =          "001101_000000001";
    decode_dict["001000001"] =          "001101_111111111";

    decode_dict["00000011100"] =        "001110_000000001";
    decode_dict["00000011101"] =        "001110_111111111";

    decode_dict["00000011010"] =        "001111_000000001";
    decode_dict["00000011011"] =        "001111_111111111";

    decode_dict["00000010000"] =        "010000_000000001";
    decode_dict["00000010001"] =        "010000_111111111";

    decode_dict["0000000111110"] =      "010001_000000001";
    decode_dict["0000000111111"] =      "010001_111111111";

    decode_dict["0000000110100"] =      "010010_000000001";
    decode_dict["0000000110101"] =      "010010_111111111";

    decode_dict["0000000110010"] =      "010011_000000001";
    decode_dict["0000000110011"] =      "010011_111111111";

    decode_dict["0000000101110"] =      "010100_000000001";
    decode_dict["0000000101111"] =      "010100_111111111";

    decode_dict["0000000101100"] =      "010101_000000001";
    decode_dict["0000000101101"] =      "010101_111111111";

    decode_dict["00000000111110"] =     "010110_000000001";
    decode_dict["00000000111111"] =     "010110_111111111";

    decode_dict["00000000111100"] =     "010111_000000001";
    decode_dict["00000000111101"] =     "010111_111111111";

    decode_dict["00000000111010"] =     "011000_000000001";
    decode_dict["00000000111011"] =     "011000_111111111";

    decode_dict["00000000111000"] =     "011001_000000001";
    decode_dict["00000000111001"] =     "011001_111111111";

    decode_dict["00000000110110"] =     "011010_000000001";
    decode_dict["00000000110111"] =     "011010_111111111";

    decode_dict["000001"] = "ESCAPE";
}