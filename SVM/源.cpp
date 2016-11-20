
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <time.h>
#include <string>
#include <algorithm>
#include <io.h>
using namespace std;
vector<vector<float>>dataMat;
vector<int>labelMat;
vector<vector<float>>dataMat2;
vector<int>labelMat2;
vector<float> kernelTrans(vector<vector<float>>X, vector<float>A, string kTup, float kTup_lev);
class optStruct
{
public:
	vector<vector<float>>X;
	vector<int>labelMat;
	float C;
	float tol;
	int m;
	vector<float>alphas;
	float b;
	vector<vector<float>> eCache;
	string kTup;
	float kTup_lev;
	vector<vector<float>>K;
	optStruct(vector<vector<float>>dataMatIn, vector<int>classlabel,float c,float toler,string tup,float tup_level=1)
	{
		X = dataMatIn;
		labelMat = classlabel;
		C = c;
		tol = toler;
		m = dataMatIn.size();
		alphas = vector<float>(m,0);
		b = 0;
		kTup = tup;
		kTup_lev = tup_level;
		K = *new vector<vector <float> >(m, vector<float>(m, 0));
		for (int i = 0; i < m; i++)
		{
			vector<float> temp_k = kernelTrans(X, X[i], kTup, kTup_lev);
			for (unsigned int j = 0; j < temp_k.size(); j++)
				K[j][i] = temp_k[j];
		}
		//vector<vector <float> > ivec(2, vector<float>(m, 0));
		eCache =* new vector<vector <float> >(2, vector<float>(m, 0));
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
vector<float> multiply_vv(vector<float>alphas, vector<int>labelMat)
{
	vector<float> res ;
	for (unsigned int i = 0; i < alphas.size(); i++)
	{
		res.push_back(alphas[i] * labelMat[i]);
	}
	return res;
}
vector<float> multiply_vv(vector<float>alphas, vector<float>labelMat)
{
	vector<float> res;
	for (unsigned int i = 0; i < alphas.size(); i++)
	{
		res.push_back(alphas[i] * labelMat[i]);
	}
	return res;
}

vector<float> multiply_mv(vector<vector<float>>mat, vector<float>vec)
{
	vector<float> res;
	for (unsigned int i = 0; i < mat.size(); i++)
	{
		float temp = 0;
		for (unsigned int j = 0; j < mat[0].size(); j++)
		{
			temp += mat[i][j] * vec[j];
		}
		res.push_back(temp);
	}
	return res;
}
vector<float> add_vv(vector<float>v1, vector<float>v2,int bei=1)
{
	vector<float> res;
	for (unsigned int i = 0; i < v1.size(); i++)
	{
		res.push_back(v1[i] + v2[i]*bei);
	}
	return res;
}
vector<float> add_vv(vector<float>v1, vector<float>v2, float bei = 1)
{
	vector<float> res;
	for (unsigned int i = 0; i < v1.size(); i++)
	{
		res.push_back(v1[i] + v2[i] * bei);
	}
	return res;
}
vector<float> sub_vv(vector<float>v1, vector<float>v2, float bei = 1)
{
	vector<float> res;
	for (unsigned int i = 0; i < v1.size(); i++)
	{
		res.push_back(v1[i] - v2[i] * bei);
	}
	return res;
}
float mul_vv(vector<float>vec1, vector<float>vec2)
{
	float res=0;
	for (unsigned int i = 0; i < vec1.size(); i++)
	{
		res+=vec1[i] * vec2[i];
	}
	return res;
}
int selectJrand(int i, int m)
{
	int j = i;
	srand((unsigned)time(NULL));
	while (j == i)
	{
		j =  rand() % m;
	}
	return j;
}

float clipAlpha(float aj, float H, float L)
{
	if (aj > H)
		aj = H;
	if (aj < L)
		aj = L;
	return aj;
}

void smoSimple(vector<vector<float>>dataMatIn, vector<int>classLabels, float C, float toler, float maxIter, float &b, vector<float> &alphas)
{
	int m = dataMatIn.size();
	int n = dataMatIn[0].size();
	int iter = 0;
	while (iter < maxIter)
	{	
		int alphaPairsChanged = 0;
		for (int i = 0; i < m; i++)
		{
			float fxi = float(mul_vv(multiply_vv(alphas, labelMat), multiply_mv(dataMatIn, dataMatIn[i])))+b;
			float Ei = fxi - float(labelMat[i]);
			if (((labelMat[i] * Ei<-toler) && (alphas[i]<C)) || ((labelMat[i] * Ei>toler) && (alphas[i]>0)))
			{
				int j = selectJrand(i, m);
				float fxj = float(mul_vv(multiply_vv(alphas, labelMat), multiply_mv(dataMatIn, dataMatIn[j]))) + b;
				float Ej = fxj - float(labelMat[j]);
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
				float eta = float(2.0)*mul_vv(dataMatIn[i], dataMatIn[j]) - mul_vv(dataMatIn[i], dataMatIn[i]) - mul_vv(dataMatIn[j], dataMatIn[j]);
				if (eta >= 0)
				{
					cout << "eta>=0" << endl;
					continue;
				}
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta;
				alphas[j] = clipAlpha(alphas[j], H, L);
				if (abs(alphas[j]-alphaJold)<0.0001)
				{
					cout << "j not moving enough" << endl;
					continue;
				}
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j]);
				float b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold)*mul_vv(dataMatIn[i], dataMatIn[i]) - labelMat[j] * (alphas[j] - alphaJold)*mul_vv(dataMatIn[i], dataMatIn[j]);
				float b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold)*mul_vv(dataMatIn[i], dataMatIn[j]) - labelMat[j] * (alphas[j] - alphaJold)*mul_vv(dataMatIn[j], dataMatIn[j]);
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

float calcEk(optStruct oS, int k)
{
	vector<float> osk;
	for (unsigned int i = 0; i < oS.K.size(); i++)
	{
		osk.push_back(oS.K[i][k]);
	}
	float fXk = float(mul_vv(multiply_vv(oS.alphas, oS.labelMat),osk))+oS.b;
	return fXk - float(oS.labelMat[k]);
}

vector<int> nonzero(vector<float> vec)
{
	vector<int>res;
	vector<float>::iterator it = vec.begin();
	int i = 0;
	for (; it != vec.end(); it++,i++)
	{
		if (*it != 0)
			res.push_back(i);
	}
	return res;
}
vector<int> nonzero(vector<float> vec,float l1,float l2)
{
	vector<int>res;
	vector<float>::iterator it = vec.begin();
	int i = 0;
	for (; it != vec.end(); it++, i++)
	{
		if ((*it <l2)&&(*it>l1))
			res.push_back(i);
	}
	return res;
}
void selectJ(int i, optStruct &oS, float Ei,int &j,float &Ej)
{
	int maxK = -1;
	float maxDeltaE = 0;
	Ej = 0;
	oS.eCache[0][i] = 1;
	oS.eCache[1][i] = Ei;
	vector<int> validEcacheList = nonzero(oS.eCache[0]);
	if (validEcacheList.size() > 1)
	{

		vector<int>::iterator k = validEcacheList.begin();
		for (; k != validEcacheList.end(); k++)
		{
			if (*k == i)
				continue;
			float Ek = calcEk(oS,*k);
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
		j =  selectJrand(i, oS.m);
		Ej = calcEk(oS, j);
	}
	return;
}

void updateEk(optStruct &oS, int k)
{
	oS.eCache[0][k] = 1;
	oS.eCache[1][k] = calcEk(oS,k);
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
		float L=0;
		float H=0;
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
		float eta = 2 * oS.K[i][j] - oS.K[i][i]-oS.K[j][j];
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
		float b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i][i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K[i][j];
		float b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i][j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K[j][j];
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
		float eta = 2 * mul_vv(oS.X[i], oS.X[j]) - mul_vv(oS.X[i], oS.X[i]) - mul_vv(oS.X[j], oS.X[j]);
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
		float b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*mul_vv(oS.X[i], oS.X[i]) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*mul_vv(oS.X[i], oS.X[j]);
		float b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*mul_vv(oS.X[i], oS.X[j]) - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*mul_vv(oS.X[j], oS.X[j]);
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

void smoP(vector<vector<float>>dataMatIn, vector<int>classLabels, float C, float toler, int maxIter, float &b, vector<float> &alphas,string kTup , float kTup_lev)
{

	optStruct oS(dataMatIn,classLabels,C,toler,kTup,kTup_lev);
	int iter = 0;
	bool entireSet = true;
	int alphaPairsChanged = 0;
	while ((iter < maxIter)&&((alphaPairsChanged>0)||(entireSet)))
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
			vector<int>nonBoundIs = nonzero(oS.alphas,0,C);
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

vector<float> calcWs(vector<float>alphas, vector<vector<float>>dataArr, vector<int>classLabels)
{
	int m = dataArr.size();
	int n = dataArr[0].size();
	vector<float>w(n, 0);
	for (int i = 0; i < m; i++)
	{
		w = add_vv(w, dataArr[i], alphas[i] * labelMat[i]);
	}
	return w;
}

vector<float> kernelTrans(vector<vector<float>>X, vector<float>A, string kTup, float kTup_lev)
{
	int m = X.size();
	int n = X[0].size();
	vector<float>K(m, 0);
	if (kTup == "lin")
	{
		K = multiply_mv(X, A);
	}
	else if (kTup == "rbf")
	{
		for (int j = 0; j < m; j++)
		{
			vector<float>deltaRow = sub_vv(X[j], A);
			K[j] = mul_vv(deltaRow, deltaRow);
		}
		for (unsigned int j = 0; j < K.size(); j++)
		{
			K[j] = exp(K[j] / (-1 * kTup_lev *kTup_lev));
		}
	}
	else
	{
		cout << "raise NameError('Houston we have a problem that kernal is not recognized')" << endl;
	}
	return K;
}

void testRbf(float k1 = 1.3)
{
	loadDataSet("testSetRBF.txt");
	vector<float>alphas(dataMat.size(), 0);
	float b = 0;
	//optStruct A(dataMat,labelMat,0.6,0.1);
	//smoSimple(dataMat,labelMat,0.6, 0.001, 40, bb, alphas1);
	smoP(dataMat, labelMat, float(200), float(0.0001),10000, b, alphas,"rbf",k1);
	vector<int>svInd = nonzero(alphas,0,10000);
	vector<vector<float>>sVs;
	vector<float>labelSV;
	vector<float>alphaSV;
	for (unsigned int i = 0; i < svInd.size(); i++)
	{
		sVs.push_back(dataMat[svInd[i]]);
		labelSV.push_back(labelMat[svInd[i]]);
		alphaSV.push_back(alphas[svInd[i]]);
	}
	cout << "there are " << sVs.size() << " support vectors"<<endl;
	int errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		vector<float>kernelEval = kernelTrans(sVs, dataMat[i], "rbf", k1);
		float predict = mul_vv(kernelEval, multiply_vv(labelSV, alphaSV)) + b;
		if (predict*labelMat[i] < 0)
		{
			cout << "got wrong" << endl;
			errorCount++;
		}
	}
	cout << "training error rate is " << float(errorCount) / dataMat.size() << endl;
	loadDataSet("testSetRBF2.txt");
	errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		vector<float>kernelEval = kernelTrans(sVs, dataMat[i], "rbf", k1);
		float predict = mul_vv(kernelEval, multiply_vv(labelSV, alphaSV)) + b;
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
	while (getline(in, line_temp)&&line<32)
	{
		line++;	
		for (int i = 0; i < 32;i++)
			back.push_back(line_temp[i]-'0');
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
			
		} while (_findnext(hFile,&fileinfo)==0);
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
	vector<float>alphas(trainImgMat.size(), 0);
	float b = 0;
	smoP(trainImgMat, trainImglab, float(200), float(0.0001), 10000, b, alphas, "rbf", k1);
	vector<int>svInd = nonzero(alphas, 0, 10000);
	vector<vector<float>>sVs;
	vector<float>labelSV;
	vector<float>alphaSV;
	for (unsigned int i = 0; i < svInd.size(); i++)
	{
		sVs.push_back(trainImgMat[svInd[i]]);
		labelSV.push_back(trainImglab[svInd[i]]);
		alphaSV.push_back(alphas[svInd[i]]);
	}
	cout << "there are " << sVs.size() << " support vectors" << endl;
	int errorCount = 0;
	for (unsigned int i = 0; i < dataMat.size(); i++)
	{
		vector<float>kernelEval = kernelTrans(sVs, trainImgMat[i], "rbf", k1);
		float predict = mul_vv(kernelEval, multiply_vv(labelSV, alphaSV)) + b;
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
		vector<float>kernelEval = kernelTrans(sVs, testImgMat[i], "rbf", k1);
		float predict = mul_vv(kernelEval, multiply_vv(labelSV, alphaSV)) + b;
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