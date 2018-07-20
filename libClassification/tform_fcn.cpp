#include "tform_fcn.h"
cv::Mat min(cv::Mat minrow){
	cv::Mat ret(1, minrow.cols, CV_32FC1);
	for (int col = 0; col < minrow.cols; col++){
		float minnum = minrow.at<float>(0, col);
		for (int row = 0; row < minrow.rows; row++){
			if (minrow.at<float>(row, col) < minnum)
				minnum = minrow.at<float>(row, col);
		}
		ret.at<float>(0, col) = minnum;
	}
	return ret;
}

cv::Mat max(cv::Mat maxrow){
	cv::Mat ret(1, maxrow.cols, CV_32FC1);
	for (int col = 0; col < maxrow.cols; col++){
		float maxnum = maxrow.at<float>(0, col);
		for (int row = 0; row < maxrow.rows; row++){
			if (maxrow.at<float>(row, col) > maxnum)
				maxnum = maxrow.at<float>(row, col);
		}
		ret.at<float>(0, col) = maxnum;
	}
	return ret;
}

cv::Mat getshift(cv::Mat mat){//二
	float tol = 1000.0f;
	cv::Mat minPoints = min(mat);
	cv::Mat maxPoints = max(mat);
	cv::Mat center = (minPoints + maxPoints) / 2;
	cv::Mat span = maxPoints - minPoints;
	if ((span.at<float>(0, 0) > 0 && abs(center.at<float>(0, 0)) / span.at<float>(0, 0) > tol) ||
		(span.at<float>(0, 1) > 0 && abs(center.at<float>(0, 1)) / span.at<float>(0, 1) > tol)){
		return center;
	}else{
		center.at<float>(0, 0) = 0;
		center.at<float>(0, 1) = 0;
		return center;
	}
}
int rank(cv::Mat src){//求秩
	//cv::Mat S = cv::Mat(src.rows, 1, CV_32FC1);
	cv::SVD thissvd(src, cv::SVD::FULL_UV);//矩阵奇异分解
	cv::Mat S = thissvd.w;
	int sum = 0;
	for (int row= 0; row < S.rows; row++){
		if (S.at<float>(row, 0)>0.00001)
			sum++;
	}
	return sum;
}


Tfm findNonreflectiveSimilarity(cv::Mat src, cv::Mat dst, options opt){//四
	int k = opt.K;
	cv::Mat X(dst.rows * 2, 4, CV_32FC1);
	for (int col = 0; col < X.cols; col++){
		for (int row = 0; row < X.rows;row++){
			if (col == 0){			
				X.at<float>(row, col) = dst.at<float>(row%dst.rows, row / dst.rows);
			}
			else if (col == 1){
				X.at<float>(row, col) = (-2 *( row / dst.rows)+1)*dst.at<float>(row%dst.rows, 1 - row / dst.rows);
			}
			else if (col == 2){
				if (row < X.rows / 2)
					X.at<float>(row, col) = 1.0f;
				else
					X.at<float>(row, col) = 0;
			}
			else if (col == 3){
				if (row < X.rows / 2)
					X.at<float>(row, col) = 0;
				else
					X.at<float>(row, col) = 1.0f;
			}
		}
	}
	cv::Mat U(src.rows * 2, 1, CV_32FC1);
	for (int col = 0; col < U.cols; col++){
		for (int row = 0; row < U.rows; row++){
			U.at<float>(row, col) = src.at<float>(row%src.rows, row / src.rows);
		}
	}
	if (rank(X) >= k * 2){		
		cv::MatExpr x = X.inv(cv::DECOMP_SVD);
		cv::Mat xx = x.operator cv::Mat();
		cv::Mat r(xx.rows,U.cols,CV_32FC1);//r = X \U ==》 xx*U
		for (int row = 0; row < r.rows; row++){
			r.at<float>(row, 0) = 0;
			for (int num = 0; num < U.rows; num++){
				r.at<float>(row, 0) += xx.at<float>(row,num) * U.at<float>(num,0);
			}
		}
		float sc = r.at<float>(0, 0);
		float ss = r.at<float>(1, 0);
		float tx = r.at<float>(2, 0);
		float ty = r.at<float>(3, 0);
		cv::Mat Tinv(3, 3, CV_32FC1);
		Tinv.at<float>(0, 0) = sc; Tinv.at<float>(0, 1) = -ss; Tinv.at<float>(0, 2) = 0;
		Tinv.at<float>(1, 0) = ss; Tinv.at<float>(1, 1) = sc; Tinv.at<float>(1, 2) = 0;
		Tinv.at<float>(2, 0) = tx; Tinv.at<float>(2, 1) = ty; Tinv.at<float>(2, 2) = 1;
		cv::Mat T = Tinv.inv(cv::DECOMP_SVD).operator cv::Mat();
		T.at<float>(0, 2) = 0; T.at<float>(1, 2) = 0; T.at<float>(2, 2) = 1;
		Tfm retfm;
		retfm.forword = T;
		retfm.inv = Tinv;
		return retfm;
	}
	else {
		throw matlabexception();
	}
}

Tfm affine(cv::Mat multrTY){
	Tfm ret;
	ret.forword = multrTY;
	ret.inv = multrTY.inv().operator cv::Mat();
	ret.inv.at<float>(0, 2) = 0; ret.inv.at<float>(1, 2) = 0; ret.inv.at<float>(2, 2) = 1;
	return ret;
}

cv::Mat tformfwd(Tfm trans1, cv::Mat U){
	cv::Mat M = trans1.forword;
	cv::Mat X1(U.rows, U.cols + 1, CV_32FC1);
	for (int col = 0; col < X1.cols;col++)
	for (int row = 0; row < X1.rows; row++){
		if (col == 2)
			X1.at<float>(row, col) = 1;
		else
			X1.at<float>(row, col) = U.at<float>(row, col);
	}
	cv::Mat U1(X1.rows,M.cols,CV_32FC1);//X1 * M
	for (int col = 0; col < U1.cols; col++){
		for (int row = 0; row < U1.rows; row++){
			U1.at<float>(row, col) = 0;
			for (int num = 0; num < X1.cols; num++){
				U1.at<float>(row, col) += X1.at<float>(row,num)*M.at<float>(num,col);
			}
		}
	}
	U1.adjustROI(0,0,0,-1);
	return U1;
}

Tfm findT_fcn(cv::Mat src, cv::Mat dst, options opt){//三
	opt.K = 2;
	Tfm train1 = findNonreflectiveSimilarity(src, dst, opt);
	cv::Mat dst2;
	dst.convertTo(dst2,CV_32FC1);
	for (int row = 0; row < dst2.rows; row++){
		dst2.at<float>(row, 0) = -1 * dst2.at<float>(row, 0);
		//std::cout << "dst2: " << dst2.at<float>(row, 0) << std::endl;
	}
	Tfm train2r = findNonreflectiveSimilarity(src, dst2, opt);
	cv::Mat TreflectY(3,3,CV_32FC1);
	TreflectY.at<float>(0, 0) = -1; TreflectY.at<float>(0, 1) = 0; TreflectY.at<float>(0,2) = 0;
	TreflectY.at<float>(1, 0) = 0; TreflectY.at<float>(1, 1) = 1; TreflectY.at<float>(1, 2) = 0;
	TreflectY.at<float>(2, 0) = 0; TreflectY.at<float>(2, 1) = 0; TreflectY.at<float>(2, 2) = 1;
	cv::Mat multrTY(train2r.forword.rows, TreflectY.cols, CV_32FC1);//trani2r.forword * TreflectY
	for (int col = 0; col < multrTY.cols; col++){
		for (int row = 0; row < multrTY.rows; row++){
			multrTY.at<float>(row, col) = 0;
			for (int tr_col = 0; tr_col < train2r.forword.cols; tr_col++){
				multrTY.at<float>(row, col) += train2r.forword.at<float>(row, tr_col) * TreflectY.at<float>( tr_col,col);				
			}
			//std::cout << multrTY.at<float>(row, col) << std::endl;
		}
	}
	Tfm train2 = affine(multrTY);
	cv::Mat xy1 = tformfwd(train1, src);
	cv::Mat sub1;
	cv::subtract(xy1,dst,sub1);//sub1 = xy1 -dst
	double norm1 = cv::norm(sub1, cv::NORM_L2);
	/*std::cout << sub1.at<float>(0, 0) << "  " << sub1.at<float>(0, 1) << std::endl;
	std::cout << sub1.at<float>(1, 0) << "  " << sub1.at<float>(1, 1) << std::endl;
	std::cout << sub1.at<float>(2, 0) << "  " << sub1.at<float>(2, 1) << std::endl;
	std::cout << sub1.at<float>(3, 0) << "  " << sub1.at<float>(3, 1) << std::endl;
	std::cout << sub1.at<float>(4, 0) << "  " << sub1.at<float>(4, 1) << std::endl;*/

	cv::Mat xy2 = tformfwd(train2, src);
	cv::Mat sub2;
	cv::subtract(xy2, dst, sub2);//sub2 = xy2 -dst
	double norm2 = cv::norm(sub2, cv::NORM_L2);
	/*std::cout << sub2.at<float>(0, 0) << "  " << sub2.at<float>(0, 1) << std::endl;
	std::cout << sub2.at<float>(1, 0) << "  " << sub2.at<float>(1, 1) << std::endl;
	std::cout << sub2.at<float>(2, 0) << "  " << sub2.at<float>(2, 1) << std::endl;
	std::cout << sub2.at<float>(3, 0) << "  " << sub2.at<float>(3, 1) << std::endl;
	std::cout << sub2.at<float>(4, 0) << "  " << sub2.at<float>(4, 1) << std::endl;*/
	if (norm1 <= norm2)
		return train1;
	else
		return train2;
}

Tfm cp2tform_similarity(cv::Mat src, cv::Mat dst){	// 一
	Tfm ret;
	options opt;
	opt.order = 3; opt.K = 3;
	int M = src.rows;
	if (M < opt.K)
		throw matlabexception();
	cv::Mat srcshift = getshift(src);
	cv::Mat dstshift = getshift(dst);
	int needToShift = 0;
	for (int row = 0; row < srcshift.rows; row++){
		for (int col = 0; col < srcshift.cols; col++){
			if (srcshift.at<float>(row, col) < -0.00001 || srcshift.at<float>(row, col) > 0.00001)
				needToShift = 1;
		}
	}
	for (int row = 0; row < dstshift.rows; row++){
		for (int col = 0; col < dstshift.cols; col++){
			if (dstshift.at<float>(row, col) < -0.00001 || dstshift.at<float>(row, col) > 0.00001)
				needToShift = 1;
		}
	}
	if (!needToShift)
		ret = findT_fcn(src, dst, opt);
	else
		throw matlabexception();
	return ret;
}


cv::Mat tform_inv(cv::Mat G, /*Tfm reg_b,*/ Tfm tm/*, Tfm reg_a*/){
	cv::Mat inv = tm.inv;
	cv::Mat G1(G.rows, G.cols, CV_32FC2); //G1 = G*inv

	for (int n = 0; n < 2;n++)
	for (int col = 0; col < G.cols; col++){
		for (int row = 0; row < G.rows; row++){
			G1.at<cv::Vec2f>(row, col)[n] = 0;
			for (int num = 0; num < inv.rows; num++){
			
				if (num < 2)
					G1.at<cv::Vec2f>(row, col)[n] += G.at<cv::Vec2f>(row, col)[num] * inv.at<float>(num, n);//x*costh+y*sinth  x*sinth+y*costh
				else
					G1.at<cv::Vec2f>(row, col)[n] += inv.at<float>(num, n);//+inv * 1
			}	
	
		}
	}
	return G1;
}
cv::Mat itransform(cv::Mat img, Tfm tm, int rows, int cols){
	/*Tfm reg_b;
	reg_b.forword = cv::Mat(1, 2, CV_32FC1);
	reg_b.forword.at<float>(0, 0) = 1; reg_b.forword.at<float>(0, 1) = 1;
	reg_b.inv = cv::Mat(1, 2, CV_32FC1);
	reg_b.inv.at<float>(0, 0) =0; reg_b.inv.at<float>(0, 1) = 0;

	Tfm reg_a;
	reg_a.forword = cv::Mat(1, 2, CV_32FC1);
	reg_a.forword.at<float>(0, 0) = 1; reg_a.forword.at<float>(0, 1) = 1;
	reg_a.inv = cv::Mat(1, 2, CV_32FC1);
	reg_a.inv.at<float>(0, 0) = 0; reg_a.inv.at<float>(0, 1) = 0;*/
	
	cv::Mat G(cols, rows, CV_32FC2);

	for (int col = 0; col < G.cols; col++){
		for (int row = 0; row < G.rows; row++){
			G.at<cv::Vec2f>(row, col)[0] = row/*+1*/;//存坐标
			G.at<cv::Vec2f>(row, col)[1] = col /*+ 1*/;
		}
	}

	cv::Mat M = tform_inv(G,/* reg_b, */tm/*, reg_a*/);

	cv::Mat MT = M.t().operator cv::Mat();//MT 112*96*2;
	/*for (int n = 0; n < 2;n++)
	for (int row = 0; row < M.rows; row++){
		for (int col = 0; col < M.cols; col++){
			std::cout << M.at<cv::Vec2f>(row, col)[n] << "  " << MT.at<cv::Vec2f>(col, row)[n] << std::endl;
		}
	}*/
	cv::Mat dst(rows, cols, img.type());
	
	for (int row = 0; row < MT.rows; row++){
		for (int col = 0; col < MT.cols; col++){
			int  x1 = (int)MT.at<cv::Vec2f>(row, col)[0];
			if (x1 < 0) x1 = 0;
			else if (x1 >= img.cols - 1) x1 = img.cols - 2;
			int x2 = x1 + 1;
			int y1 = (int)MT.at<cv::Vec2f>(row, col)[1];
			if (y1 < 0) y1 = 0;
			else if (y1 >= img.rows - 1) y1 = img.rows - 2;
			int y2 = y1 + 1;

			float x1_f = x2 - MT.at<cv::Vec2f>(row, col)[0];
			float x2_f =1- x1_f; 
			float y1_f = y2 - MT.at<cv::Vec2f>(row, col)[1];
			float y2_f = 1 - y1_f;
			for (int n = 0; n < dst.channels(); n++){
				dst.at<cv::Vec3b>(row, col)[n] = int(img.at<cv::Vec3b>(y1, x1)[n] * y1_f*x1_f + img.at<cv::Vec3b>(y2, x2)[n] * y2_f*x2_f + img.at<cv::Vec3b>(y2, x1)[n] * y2_f*x1_f + img.at<cv::Vec3b>(y1, x2)[n] * y1_f*x2_f);
			}
		}
	}
	return dst;
}


//旋转图像内容不变，尺寸相应变大loose
cv::Mat  rotateImage1(cv::Mat img, double degree)
{
	double angle = degree  * CV_PI / 180.; // 弧度  
	double a = sin(angle), b = cos(angle);
	int width = img.cols;
	int height = img.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));

	// 旋转中心
	cv::Point center = cv::Point(width / 2, height / 2);
	cv::Mat map_Matrix = cv::getRotationMatrix2D(center, degree, 1.0);

	map_Matrix.at<double>(0, 2) += (width_rotate - width) / 2;
	map_Matrix.at<double>(1, 2) += (height_rotate - height) / 2;

	cv::Mat img_rotate;
	// INTER_NEARST,INTER_LINEAR ,CV_INTER_CUBIC
	cv::warpAffine(img, img_rotate, map_Matrix, { width_rotate, height_rotate }, cv::INTER_CUBIC | CV_WARP_FILL_OUTLIERS, cv::BORDER_CONSTANT, cv::Scalar());
	return img_rotate;
}
//face h:w == 1:1
cv::Mat onetone(cv::Mat img, cv::Mat point, int size, int emh, int eh){
	if (img.empty() || point.empty()) throw(matlabexception());
	cv::Mat F3;
	img.convertTo(F3, CV_32FC3);
	double ang_tan = 1.0*(point.at<float>(0, 1) - point.at<float>(1, 1)) / (point.at<float>(0, 0) - point.at<float>(1, 0));//dy/dx 
 	float ang = atan(ang_tan) / CV_PI * 180;//角度
	cv::Mat img_rot = rotateImage1(F3, ang);//旋转
	//int imgh = img.rows;
	//int imgw = img.cols;
	//eye_ceneter
	double eye_center_x = 1.0*(point.at<float>(0, 0) + point.at<float>(1, 0)) / 2;
	double eye_center_y = 1.0*(point.at<float>(0, 1) + point.at<float>(1, 1)) / 2;
	ang = -ang / 180 * CV_PI;//弧度

	double x0 = eye_center_x - 1.0*img.cols / 2;
	double y0 = eye_center_y - 1.0*img.rows / 2;

	int eye_center_xx = round(x0*cos(ang) - y0*sin(ang) + 1.0*img_rot.cols / 2);
	int eye_center_yy = round(x0*sin(ang) + y0*cos(ang) + 1.0*img_rot.rows / 2);//eye center

	double mth_center_x = 1.0*(point.at<float>(3, 0) + point.at<float>(4, 0)) / 2;
	double mth_center_y = 1.0*(point.at<float>(3, 1) + point.at<float>(4, 1)) / 2;

	double x00 = mth_center_x - 1.0*img.cols / 2;
	double y00 = mth_center_y - 1.0*img.rows / 2;

	int mth_center_xx = round(x00*cos(ang) - y00*sin(ang) + 1.0*img_rot.cols / 2);
	int mth_center_yy = round(x00*sin(ang) + y00*cos(ang) + 1.0*img_rot.rows / 2);//mouth center

	double re_scal = 1.0* emh / (mth_center_yy - eye_center_yy);
	int imgh = img_rot.rows;
	int imgw = img_rot.cols;
	cv::Mat img_rot_resize;
	cv::resize(img_rot, img_rot_resize, cv::Size(img_rot.cols*re_scal, img_rot.rows*re_scal));


	int eye_new_x = round((eye_center_xx - 1.0*imgw / 2)*re_scal + 1.0*img_rot_resize.cols / 2);
	int eye_new_y = round((eye_center_yy - 1.0*imgh / 2)*re_scal + 1.0*img_rot_resize.rows / 2);

	cv::Mat img_crop(cv::Size(size, size), img_rot_resize.type());

	int crop_y = eye_new_y - eh;
	int crop_y_end = crop_y + size - 1;
	int cropy_x = eye_new_x - floor(1.0*size / 2);
	int cropy_x_end = cropy_x + size - 1;

	int box_x = cropy_x, box_x_end = cropy_x_end, box_y = crop_y, box_y_end = crop_y_end;
	if (cropy_x < 1) box_x = 1;
	if (cropy_x > img_rot_resize.cols) box_x = img_rot_resize.cols;
	if (cropy_x_end < 1) box_x_end = 1;
	if (cropy_x_end > img_rot_resize.cols) box_x_end = img_rot_resize.cols;

	if (crop_y < 1) box_y = 1;
	if (crop_y > img_rot_resize.rows) box_y = img_rot_resize.rows;
	if (crop_y_end < 1) box_y_end = 1;
	if (crop_y_end > img_rot_resize.rows) box_y_end = img_rot_resize.rows;

	for (int h = box_y - crop_y; h < box_y_end - crop_y; h++){
		for (int w = box_x - cropy_x; w < box_x_end - cropy_x; w++){
			img_crop.at<cv::Vec3f>(h, w)[0] = img_rot_resize.at<cv::Vec3f>(h + crop_y, w + cropy_x)[0];
			img_crop.at<cv::Vec3f>(h, w)[1] = img_rot_resize.at<cv::Vec3f>(h + crop_y, w + cropy_x)[1];
			img_crop.at<cv::Vec3f>(h, w)[2] = img_rot_resize.at<cv::Vec3f>(h + crop_y, w + cropy_x)[2];
		}
	}
	//cv::Mat ret;
	//img_crop.convertTo(ret, CV_8UC3);
	return img_crop;
}