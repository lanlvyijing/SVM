/*12345*/
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <time.h>
#include <string>
#include <algorithm>
#include <io.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
vector<vector<float>>dataMat;
vector<int>labelMat;
VectorXf kernelTrans(MatrixXf X, VectorXf A, string kTup, float kTup_lev)
{

	int m = X.rows();
	int n = X.cols();
	VectorXf K(m,1);
	if (kTup == "lin")
	{
		K = X * (A.transpose());
	}
	else if (kTup == "rbf")
	{
		for (int j = 0; j < m; j++)
		{
			VectorXf deltaRow = X.row(j) - A.transpose();
			K[j] = deltaRow.dot(deltaRow);
		}
		float ktup_temp = (-1 * kTup_lev *kTup_lev);
		for (unsigned int j = 0; j < K.size(); j++)
		{
			K[j] = exp(K[j] / ktup_temp);
		}
	}
	else
	{
		cout << "raise NameError('Houston we have a problem that kernal is not recognized')" << endl;
	}
	return K;
}
class optStruct
{
public:
	MatrixXf X;
	VectorXf labelMat;
	float C;
	float tol;
	int m;
	VectorXf alphas;
	float b;
	MatrixXf eCache;
	string kTup;
	float kTup_lev;
	MatrixXf K;
	optStruct(MatrixXf dataMatIn, VectorXf classlabel, float c, float toler, string tup, float tup_level = 1)
	{
		X = dataMatIn;
		labelMat = classlabel;
		C = c;
		tol = toler;
		m = dataMatIn.rows();
		alphas = VectorXf::Zero(m);
		b = 0;
		kTup = tup;
		kTup_lev = tup_level;
		K = MatrixXf::Zero(m,m);
		for (int i = 0; i < m; i++)
		{
			VectorXf temp_k = kernelTrans(X, X.row(i), kTup, kTup_lev);
			
			K.col(i) = temp_k.transpose();
		}
		eCache = MatrixXf::Zero(2, m);
	}
};

void loadDataSet(string filename)
{
	ifstream in;
	string line_temp;
	int line = 0;
	in.open(filename, ios::in);//ios::in 表示以只读的方式读取文件
	while (getline(in, line_temp))
	{
		line++;
	}
	in.close();
	in.open(filename, ios::in);
	for (int i = 0; i < line; i++)
	{
		vector<float>temp_mat_line;
		for (int j = 0; j < 2; j++)
		{
			float temp_mat;
			in >> temp_mat;
			temp_mat_line.push_back(temp_mat);
			in.get();
		}
		dataMat.push_back(temp_mat_line);
		float temp_mat;
		in >> temp_mat;
		labelMat.push_back(int(temp_mat));
		in.get();
	}
	in.close();	
}
float calcEk(optStruct oS, int k)
{
	float fXk = (oS.alphas.cwiseProduct(oS.labelMat)).dot(oS.K.col(k)) + oS.b;
	return fXk - float(oS.labelMat[k]);
}
vector<int> nonzero(VectorXf vec)
{
	vector<int>res;
	for (int i = 0; i < vec.size();  i++)
	{
		if (vec[i] != 0)
			res.push_back(i);
	}
	return res;
}
vector<int> nonzero(VectorXf vec,float l1,float l2)
{
	vector<int>res;
	for (int i = 0; i < vec.size(); i++)
	{
		if (vec[i] <l2&&vec[i]>l1)
			res.push_back(i);
	}
	return res;
}
float clipAlpha(float aj, float H, float L)
{
	if (aj > H)
		aj = H;
	if (aj < L)
		aj = L;
	return aj;
}
int selectJrand(int i, int m)
{
	int j = i;
	srand((unsigned)time(NULL));
	while (j == i)
	{
		j = rand() % m;
	}
	return j;
}
void selectJ(int i, optStruct &oS, float Ei, int &j, float &Ej)
{
	int maxK = -1;
	float maxDeltaE = 0;
	Ej = 0;
	oS.eCache(0,i) = 1;
	oS.eCache(1,i) = Ei;
	vector<int> validEcacheList = nonzero(oS.eCache.row(0));
	if (validEcacheList.size() > 1)
	{

		vector<int>::iterator k = validEcacheList.begin();
		for (; k != validEcacheList.end(); k++)
		{
			if (*k == i)
				continue;
			float Ek = calcEk(oS, *k);
			float deltaE = abs(Ei - Ek);
			if (deltaE > maxDeltaE)
			{
				maxK = *k;
				maxDeltaE = deltaE;
				Ej = Ek;
			}
		}
		j = maxK;
	}
	else
	{
		j = selectJrand(i, oS.m);
		Ej = calcEk(oS, j);
	}
	return;
}
void updateEk(optStruct &oS, int k)
{
	oS.eCache(0,k) = 1;
	oS.eCache(1,k) = calcEk(oS, k);
}

int innerL2(int i, optStruct &oS)
{
	float Ei = calcEk(oS, i);
	if (((oS.labelMat[i] * Ei<-oS.tol) && (oS.alphas[i] <oS.C)) || ((oS.labelMat[i] * Ei>oS.tol) && (oS.alphas[i] >0)))
	{
		int j;
		float Ej;
		selectJ(i, oS, Ei, j, Ej);
		float alphaIold = oS.alphas[i];
		float alphaJold = oS.alphas[j];
		float L = 0;
		float H = 0;
		if (oS.labelMat[i] != oS.labelMat[j])
		{
			
			L = max(float(0), oS.alphas[j] - oS.alphas[i]);
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i]);
		}
		else
		{
			L = max(float(0), oS.alphas[j] + oS.alphas[i] - oS.C);
			H = min(oS.C, oS.alphas[j] + oS.alphas[i]);
		}
		if (L == H)
		{
			cout << "L==H" << endl;
			return 0;
		}
		float eta = 2 * oS.K(i,j) - oS.K(i,i) - oS.K(j,j);
		if (eta >= 0)
		{
			cout << "eta>=0" << endl;
			return 0;
		}
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta;
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L);
		updateEk(oS, j);
		if (abs(oS.alphas[j] - alphaJold) < 0.00001)
		{
			cout << "j not moving enough" << endl;
			return 0;
		}
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j]);
		updateEk(oS, i);
		float b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K(i,i) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K(i,j);
		float b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K(i,j) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K(j,j);
		if ((0 < oS.alphas[i]) && (oS.C > oS.alphas[i]))
			oS.b = b1;
		else if ((0 < oS.alphas[j]) && (oS.C > oS.alphas[j]))
			oS.b = b2;
		else
			oS.b = (b1 + b2) / float(2.0);
		return 1;
	}
	else
		return 0;
}

int innerL(int i, optStruct &oS)
{
	float Ei = calcEk(oS, i);
	if (((oS.labelMat[i] * Ei<-oS.tol) && (oS.alphas[i] <oS.C)) || ((oS.labelMat[i] * Ei>oS.tol) && (oS.alphas[i] >0)))
	{
		int j;
		float Ej;
		selectJ(i, oS, Ei, j, Ej);
		float alphaIold = oS.alphas[i];
		float alphaJold = oS.alphas[j];
		float L = 0;
		float H = 0;
		if (oS.labelMat[i] != oS.labelMat[j])
		{
			L = max(float(0), oS.alphas[j] - oS.alphas[i]);
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i]);
		}
		else
		{
			L = max(float(0), oS.alphas[j] + oS.alphas[i] - oS.C);
			H = min(oS.C, oS.alphas[j] + oS.alphas[i]);
		}
		if (L == H)
		{
			cout << "L==H" << endl;
			return 0;
		}
		float eta = 2 * oS.X.row(i).dot(oS.X.row(j)) - oS.X.row(i).dot(oS.X.row(i)) - oS.X.row(j).dot(oS.X.row(j));
		if (eta >= 0)
		{
			cout << "eta>=0" << endl;
			return 0;
		}
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta;
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L);
		updateEk(oS, j);
		if (abs(oS.alphas[j] - alphaJold) < 0.00001)
		{
			cout << "j not moving enough" << endl;
			return 0;
		}
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j]);
		updateEk(oS, i);
		float b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*(oS.X.row(i).dot(oS.X.row(i))) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*(oS.X.row(i).dot(oS.X.row(j)));
		float b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*(oS.X.row(i).dot(oS.X.row(j))) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*(oS.X.row(j).dot(oS.X.row(j)));
		if ((0 < oS.alphas[i]) && (oS.C > oS.alphas[i]))
			oS.b = b1;
		else if ((0 < oS.alphas[j]) && (oS.C > oS.alphas[j]))
			oS.b = b2;
		else
			oS.b = (b1 + b2) / float(2.0);
		return 1;
	}
	else
		return 0;
}
void smoSimple(float C, float toler, float maxIter, float &b, VectorXf &alphas)
{
	int m = dataMat.size();
	int n = dataMat[0].size();
	MatrixXf dat_m(m, n);
	VectorXf lab_v(m);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			dat_m(i, j) = dataMat[i][j];
		}
		lab_v(i) = float(labelMat[i]);
	}
	int iter = 0;
	while (iter < maxIter)
	{	
		int alphaPairsChanged = 0;
		for (int i = 0; i < m; i++)
		{

			float fxi =  (alphas.cwiseProduct(lab_v)).dot((dat_m*(dat_m.row(i).transpose()))) + b;
			float Ei = fxi - float(lab_v(i));
			if (((lab_v[i] * Ei<-toler) && (alphas[i]<C)) || ((lab_v[i] * Ei>toler) && (alphas[i]>0)))
			{
				int j = selectJrand(i, m);
				float fxj = (alphas.cwiseProduct(lab_v)).dot((dat_m*(dat_m.row(j).transpose()))) + b;
				float Ej = fxj - float(lab_v[j]);
				float alphaIold = alphas[i];
				float alphaJold = alphas[j];
				float L, H;
				if (labelMat[i] != labelMat[j])
				{
					L = max(float(0.0), alphas[j] - alphas[i]);
					H = min(C, C+alphas[j] - alphas[i]);
				}
				else
				{
					L = max(float(0.0), alphas[j] + alphas[i] - C);
					H = min(C , alphas[j] + alphas[i]);
				}
				if (L == H)
				{
					cout << "L = H = " <<L<< endl;
					continue;
				}
				float eta = float(2.0)*dat_m.row(i).dot( dat_m.row(j)) - dat_m.row(i).dot( dat_m.row(i)) -dat_m.row(j).dot( dat_m.row(j));
				if (eta >= 0)
				{
					cout << "eta>=0" << endl;
					continue;
				}
				alphas[j] -= lab_v[j] * (Ei - Ej) / eta;
				alphas[j] = clipAlpha(alphas[j], H, L);
				if (abs(alphas[j]-alphaJold)<0.0001)
				{
					cout << "j not moving enough" << endl;
					continue;
				}
				alphas[i] += lab_v[j] * lab_v[i] * (alphaJold - alphas[j]);
				float b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold)*(dat_m.row(i).dot(dat_m.row(i))) - labelMat[j] * (alphas[j] - alphaJold)*(dat_m.row(i).dot(dat_m.row(j)));
				float b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold)*(dat_m.row(i).dot(dat_m.row(j))) - labelMat[j] * (alphas[j] - alphaJold)*(dat_m.row(j).dot(dat_m.row(j)));
				if ((0 < alphas[i]) && (C > alphas[i]))
					b = b1;
				else if ((0 < alphas[j]) && (C > alphas[j]))
					b = b2;
				else
					b = (b1 + b2) / float(2.0);
				alphaPairsChanged += 1;
				cout << "iter: " << iter << " i: " << i << " pairs changed: " << alphaPairsChanged << endl;
			}
		}
		if (alphaPairsChanged == 0)
			iter += 1;
		else
			iter = 0;
		cout << "iteration number : " << iter << endl;
	}
}


void smoP(MatrixXf dataMatIn, VectorXf classLabels, float C, float toler, int maxIter, float &b, VectorXf &alphas, string kTup="lin", float kTup_lev=1.3)
{

	optStruct oS(dataMatIn, classLabels, C, toler, kTup, kTup_lev);
	int iter = 0;
	bool entireSet = true;
	int alphaPairsChanged = 0;
	while ((iter < maxIter) && ((alphaPairsChanged>0) || (entireSet)))
	{
		alphaPairsChanged = 0;
		if (entireSet)
		{
			for (int i = 0; i < oS.m; i++)
			{
				alphaPairsChanged += innerL2(i, oS);
				cout << "fullSet,iter: " << iter << " i: " << i << " pairs changed: " << alphaPairsChanged << endl;
			}
			iter++;
		}

		else
		{
			vector<int>nonBoundIs = nonzero(oS.alphas, 0, C);
			vector<int>::iterator it = nonBoundIs.begin();
			for (; it != nonBoundIs.end(); it++)
			{
				alphaPairsChanged += innerL2(*it, oS);
				cout << "non-bound,iter: " << iter << " i: " << *it << " pairs changed: " << alphaPairsChanged << endl;
			}
			iter++;
		}
		if (entireSet)
			entireSet = false;
		else
		{
			if (alphaPairsChanged == 0)
				entireSet = true;
		}
		cout << "iteration number: " << iter << endl;
	}
	b = oS.b;
	alphas = oS.alphas;
}

VectorXf calcWs(VectorXf alphas, MatrixXf dataArr, VectorXf classLabels)
{
	int m = dataArr.rows();
	int n = dataArr.cols();
	VectorXf w = VectorXf::Zero(n);
	for (int i = 0; i < m; i++)
	{
		w += dataArr.row(i)*(alphas[i] * classLabels[i]);
	}
	return w;
}

void testRbf(float k1 = 1.3)
{
	loadDataSet("testSetRBF.txt");
	float b=0;
	VectorXf alphas;	
	alphas = VectorXf::Zero(labelMat.size()) ;
	int m = dataMat.size();
	int n = dataMat[0].size();
	MatrixXf dat_m(m, n);
	VectorXf lab_v(m);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			dat_m(i, j) = dataMat[i][j];
		}
		lab_v(i) = float(labelMat[i]);
	}
	
	//optStruct A(dataMat,labelMat,0.6,0.1);
	//smoSimple(dataMat,labelMat,0.6, 0.001, 40, bb, alphas1);
	smoP(dat_m, lab_v, float(200), float(0.0001), 10000, b, alphas, "rbf", k1);
	vector<int>svInd = nonzero(alphas, 0, 10000);
	MatrixXf sVs(svInd.size(),dat_m.cols());
	VectorXf labelSV(svInd.size());
	VectorXf alphaSV(svInd.size());
	for (unsigned int i = 0; i < svInd.size(); i++)
	{
		sVs.row(i) = dat_m.row(svInd[i]);
		labelSV[i] = labelMat[svInd[i]];
		alphaSV[i] = alphas[svInd[i]];
	}
	cout << "there are " << sVs.rows() << " support vectors" << endl;
	int errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		VectorXf kernelEval = kernelTrans(sVs, dat_m.row(i), "rbf", k1);
		float predict = kernelEval.dot(labelSV.cwiseProduct(alphaSV)) + b;
		if (predict*labelMat[i] < 0)
		{
			cout << "got wrong" << endl;
			errorCount++;
		}
	}
	cout << "training error rate is " << float(errorCount) / dataMat.size() << endl;
	dataMat.clear();
	labelMat.clear();
	loadDataSet("testSetRBF2.txt");
	m = dataMat.size();
	n = dataMat[0].size();
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			dat_m(i, j) = dataMat[i][j];
		}
		lab_v(i) = float(labelMat[i]);
	}
	errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		VectorXf kernelEval = kernelTrans(sVs, dat_m.row(i), "rbf", k1);
		float predict = kernelEval.dot(labelSV.cwiseProduct(alphaSV)) + b;
		if (predict*labelMat[i] < 0)
		{
			cout << "got wrong" << endl;
			errorCount++;
		}
	}
	cout << "testing error rate is " << float(errorCount) / dataMat.size() << endl;

}


vector<float>img2vector(string filename)
{
	ifstream in;
	string line_temp;
	int line = 0;
	vector<float>back;
	in.open(filename, ios::in);//ios::in 表示以只读的方式读取文件
	while (getline(in, line_temp) && line<32)
	{
		line++;
		for (int i = 0; i < 32; i++)
			back.push_back(line_temp[i] - '0');
		in.get();
	}
	in.close();
	return back;
}
void loadImages(vector<vector<float>>&trainImgMat, vector<int> &trainImglab, vector<vector<float>>&testImgMat, vector<int> &testImglab)
{
	_finddata_t fileinfo;
	string trainingFileList = "trainingDigits";
	long hFile = 0;
	vector<string>imglist;
	if ((hFile = _findfirst("trainingDigits\\*", &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib&_A_SUBDIR))
				continue;
			else
			{
				imglist.push_back(fileinfo.name);
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	for (int i = 0; i < imglist.size(); i++)
	{
		trainImglab.push_back(atoi(imglist[i].substr(0, imglist[i].find_first_of('_')).c_str()));
		trainImgMat.push_back(img2vector("trainingDigits//" + imglist[i]));
	}

	string testFileList = "testDigits";
	hFile = 0;
	vector<string>testlist;
	if ((hFile = _findfirst("testDigits\\*", &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib&_A_SUBDIR))
				continue;
			else
			{
				testlist.push_back(fileinfo.name);
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	for (int i = 0; i < testlist.size(); i++)
	{
		testImglab.push_back(atoi(testlist[i].substr(0, testlist[i].find_first_of('_')).c_str()));
		testImgMat.push_back(img2vector("testDigits//" + testlist[i]));
	}
	return;
}
void testDigits(float k1 = 10)
{
	vector<vector<float>>trainImgMat;
	vector<int> trainImglab;
	vector<vector<float>>testImgMat;
	vector<int>testImglab;
	loadImages(trainImgMat, trainImglab, testImgMat, testImglab);
	float b = 0;
	VectorXf alphas;
	alphas = VectorXf::Zero(trainImgMat.size());
	int m = trainImgMat.size();
	int n = trainImgMat[0].size();
	MatrixXf dat_m_train(m, n);
	VectorXf lab_v_train(m);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			dat_m_train(i, j) = trainImgMat[i][j];
		}
		lab_v_train(i) = float(trainImglab[i]);
	}
	MatrixXf dat_m_test(testImgMat.size(), testImgMat[0].size());
	VectorXf lab_v_test(testImgMat.size());
	for (int i = 0; i < testImgMat.size(); i++)
	{
		for (int j = 0; j < n; j++)
		{
			dat_m_test(i, j) = testImgMat[i][j];
		}
		lab_v_test(i) = float(trainImglab[i]);
	}
	smoP(dat_m_train, lab_v_train, float(200), float(0.0001), 10000, b, alphas, "rbf", k1);
	vector<int>svInd = nonzero(alphas, 0, 10000);
	MatrixXf sVs(svInd.size(), dat_m_train.cols());
	VectorXf labelSV(svInd.size());
	VectorXf alphaSV(svInd.size());
	for (unsigned int i = 0; i < svInd.size(); i++)
	{
		sVs.row(i) = dat_m_train.row(svInd[i]);
		labelSV[i] = labelMat[svInd[i]];
		alphaSV[i] = alphas[svInd[i]];
	}
	cout << "there are " << sVs.size() << " support vectors" << endl;
	int errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		VectorXf kernelEval = kernelTrans(sVs, dat_m_train.row(i), "rbf", k1);
		float predict = kernelEval.dot(labelSV.cwiseProduct(alphaSV)) + b;
		if (predict*trainImglab[i] < 0)
		{
			cout << "got wrong" << endl;
			errorCount++;
		}
	}
	cout << "training error rate is " << float(errorCount) / trainImgMat.size() << endl;
	errorCount = 0;
	for (unsigned int i = 0; i <testImgMat.size(); i++)
	{
		VectorXf kernelEval = kernelTrans(sVs, dat_m_test.row(i), "rbf", k1);
		float predict = kernelEval.dot(labelSV.cwiseProduct(alphaSV)) + b;
		if (predict*testImglab[i] < 0)
		{
			cout << "got wrong" << endl;
			errorCount++;
		}
	}
	cout << "testing error rate is " << float(errorCount) / testImglab.size() << endl;

}

void main()
{
	testDigits();
	cin.get();

}