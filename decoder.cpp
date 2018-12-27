#include <iostream>
#include <fstream>
#include <bitset>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ASCEND true
#define DESCEND false

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

void decodeFixedBlock(Mat& block, ifstream& ifs) {
    int x = 0, y = 0;
    bool flag = ASCEND;
    // read code of the block
    string code;
    ifs >> code;

    // decode the run value
    while (!code.empty()) {
        // simulate the zigzag order
        int run = bitset<6>(code.substr(0, 6)).to_ulong();
        code.erase(0, 6);
        while (run--) {
            zigzagStep(x, y, flag);
        }
        // assign the value
        int value = bitset<9>(code.substr(0, 9)).to_ulong();
        code.erase(0, 9);
        value = value > 256 ? value - 512 : value;
        block.at<float_t>(x, y) = value;
        zigzagStep(x, y, flag);
    }
}

void iframeDecode(int num, Mat& cache_img) {
    cout << "***** Decoding picture " << num << ". *****" << endl;
    // load code
    stringstream ss;
    ss << num;
    string number = ss.str();
    while (number.length() < 4)
        number.insert(0, 1, '0');
    string read_filename = "code/" + number + ".txt";
    ifstream ifs;
    ifs.open(read_filename.c_str(), ifstream::in);

    // load PN, PL, PW
    string PN, PL, PW;
    ifs >> PN >> PL >> PW;
    int img_num = bitset<8>(PN).to_ulong();
    int img_cols = bitset<10>(PL).to_ulong();
    int img_rows = bitset<10>(PW).to_ulong();
    Mat img = Mat::zeros(Size(img_cols, img_rows), CV_8UC3);

    // extract macroblock
    int mb_rows = img_rows / 16;
    int mb_cols = img_cols / 16;
    int mb_count = mb_rows * mb_cols;
    cout << "mb rows: " << mb_rows << endl;
    cout << "mb cols: " << mb_cols << endl;

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

        // decode coeffient
        if (cbp & 32)
            decodeFixedBlock(quant_y_1, ifs);
        if (cbp & 16)
            decodeFixedBlock(quant_y_2, ifs);
        if (cbp & 8)
            decodeFixedBlock(quant_y_3, ifs);
        if (cbp & 4)
            decodeFixedBlock(quant_y_4, ifs);
        if (cbp & 2)
            decodeFixedBlock(quant_cb, ifs);
        if (cbp & 1)
            decodeFixedBlock(quant_cr, ifs);

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

        // reconstruct image macro block
        int row = 16 * (mn / mb_cols);
        int col = 16 * (mn % mb_cols);
        cout << "macro block row: " << row << ", col: " << col << endl;
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

    ifs.close();

    // convert the color space from YCrCb to RGB
    cv::cvtColor(img, img, cv::COLOR_YCrCb2RGB);

    // write into file
    string output_filename = "rebuild/" + number + ".jpg";
    imwrite(output_filename, img);

    // save img to cache img
    img.copyTo(cache_img);
}

int main(void) {
    Mat cache_img;

    iframeDecode(1, cache_img);

    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", cache_img);
    waitKey(0);

    return 0;
}
