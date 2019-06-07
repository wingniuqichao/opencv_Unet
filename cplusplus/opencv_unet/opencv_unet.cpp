// opencv_unet.cpp : 采用opencv调用unet训练模型
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include<opencv2/imgproc/imgproc.hpp>


int main()
{
	std::string modelTxt = "unet.prototxt";
	std::string modelBin = "unet.caffemodel";

	cv::dnn::Net net;
	try {
		net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	catch (cv::Exception &ee) {
		system("PAUSE");
		return -1;
	}

	cv::VideoCapture capture("test/cxk.mp4");
	if (!capture.isOpened()) {
		printf("could not load camera...\n");
		system("PAUSE");
		return -1;
	}


	cv::namedWindow("原始", cv::WINDOW_NORMAL);
	cv::namedWindow("黑白", cv::WINDOW_NORMAL);
	cv::namedWindow("扣取", cv::WINDOW_NORMAL);
	cv::Mat frame;
	cv::Mat inputBlob;
	int cnt = 0;

	while (capture.read(frame)) {
		if (frame.empty()){
			break;
		}

		cv::resize(frame, frame, cv::Size(256, 256));

		cv::imshow("原始", frame);
		//将图像做完标准化处理，转成blob后传入网络，得到输出
		cv::dnn::blobFromImage(frame, inputBlob, 1 / 255.0, cv::Size(256, 256), cv::Scalar(127.5, 127.5, 127.5), false);
		net.setInput(inputBlob, "data");
		cv::Mat detection = net.forward("predict");

		//计算耗时
		std::vector<double> layersTimings;
		double freq = cv::getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		std::cout << "FPS: " << 1000 / time << " ; time: " << time << " ms" << std::endl;
		//得到的输出是一个四维的mat格式数据，大小为[1，2, 256, 256]
		//首先将他reshape，设置成一通道，512行，256列，其中前256行与后256行是互补关系，对应位置相加都为1
		//前256行为背景的概率，后256行为人像的概率
		cv::Mat newMat = detection.reshape(1, 512);
		//获取人像概率矩阵
		newMat = newMat.rowRange(256, 512);

		cv::Mat result;
		newMat.convertTo(result, CV_8U, 255.0);

		cv::threshold(result, result, 127, 255, cv::THRESH_BINARY);
		cv::Mat result2(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
		frame.copyTo(result2, result);
		cv::imshow("黑白", result);
		cv::imshow("扣取", result2);
		cv::waitKey(1);
	}

	system("PAUSE");

    return 0;
}

