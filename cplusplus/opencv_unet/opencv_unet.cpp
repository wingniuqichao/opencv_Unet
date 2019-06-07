// opencv_unet.cpp : ����opencv����unetѵ��ģ��
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


	cv::namedWindow("ԭʼ", cv::WINDOW_NORMAL);
	cv::namedWindow("�ڰ�", cv::WINDOW_NORMAL);
	cv::namedWindow("��ȡ", cv::WINDOW_NORMAL);
	cv::Mat frame;
	cv::Mat inputBlob;
	int cnt = 0;

	while (capture.read(frame)) {
		if (frame.empty()){
			break;
		}

		cv::resize(frame, frame, cv::Size(256, 256));

		cv::imshow("ԭʼ", frame);
		//��ͼ�������׼������ת��blob�������磬�õ����
		cv::dnn::blobFromImage(frame, inputBlob, 1 / 255.0, cv::Size(256, 256), cv::Scalar(127.5, 127.5, 127.5), false);
		net.setInput(inputBlob, "data");
		cv::Mat detection = net.forward("predict");

		//�����ʱ
		std::vector<double> layersTimings;
		double freq = cv::getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		std::cout << "FPS: " << 1000 / time << " ; time: " << time << " ms" << std::endl;
		//�õ��������һ����ά��mat��ʽ���ݣ���СΪ[1��2, 256, 256]
		//���Ƚ���reshape�����ó�һͨ����512�У�256�У�����ǰ256�����256���ǻ�����ϵ����Ӧλ����Ӷ�Ϊ1
		//ǰ256��Ϊ�����ĸ��ʣ���256��Ϊ����ĸ���
		cv::Mat newMat = detection.reshape(1, 512);
		//��ȡ������ʾ���
		newMat = newMat.rowRange(256, 512);

		cv::Mat result;
		newMat.convertTo(result, CV_8U, 255.0);

		cv::threshold(result, result, 127, 255, cv::THRESH_BINARY);
		cv::Mat result2(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
		frame.copyTo(result2, result);
		cv::imshow("�ڰ�", result);
		cv::imshow("��ȡ", result2);
		cv::waitKey(1);
	}

	system("PAUSE");

    return 0;
}

