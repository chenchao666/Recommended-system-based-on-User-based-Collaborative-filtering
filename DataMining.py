from numpy import *
import warnings
from numpy import round
from numpy import savetxt
#created by chenchao, Ph.D student in zhejiang university chench@zju.edu.cn
#created in 2017-06-15

#global setting
warnings.filterwarnings("ignore")


#transform the data to the standard score matrix
def DataPrepare():
    Data=zeros((943,1682))
    txtData=loadtxt('train_all_txt.txt')
    for i in range(txtData.shape[0]):
        ind1=txtData[i,0]-1
        ind2=txtData[i,1]-1
        rank=txtData[i,2]
        Data[ind1,ind2]=rank
    save('Data.npy',Data)


def CalculateDist(user1,user2,mode):   #calculate the similarity between any two users, we provided four methods to calculate the similarity
    ind1=where(user1>0)
    ind1=ind1[0]
    ind2=where(user2>0)
    ind2=ind2[0]
    ind=intersect1d(ind1,ind2)  #find the item that both of the user1 and user2 give the score
    user1_score=user1[ind]
    user2_score=user2[ind]
    if mode=='pearson':  #pearson coefficient
        correlation=corrcoef(user1_score,user2_score)
        dist=correlation[0,1]
        if len(ind)<10:     #if there are little common items for the two users, the similarity may not precise
            dist=0
    elif mode=='cosine':  #cosine similarity
         dist=dot(user1_score,user2_score)/(linalg.norm(user1_score)*linalg.norm(user2_score))
         if len(ind)<12:
            dist=0
    elif mode=='adcosine':   #adjusted cosine similarity
        user1_score=user1_score-mean(user1_score)
        user2_score=user2_score-mean(user2_score)
        dist = dot(user1_score, user2_score) / (linalg.norm(user1_score) * linalg.norm(user2_score))
        if len(ind) < 8:
            dist = 0
    if mode=='adpearson':  #adjusted pearson coefficient
        correlation=corrcoef(user1_score,user2_score)
        dist=correlation[0,1]
        rate=2*len(ind)/(len(ind1)+len(ind2))
        dist=dist*rate
        if len(ind)<8:     #if there are little common items for the two users, the similarity may not precise
            dist=0
    return dist



#calculate the similarity between any pair of users
def SimilarityMat(Data):
    Similarity=zeros((Data.shape[0],Data.shape[0]))
    for i in range(Data.shape[0]):
        user1=Data[i,:]
        for j in range(Data.shape[0]):
            user2=Data[j,:]
            Similarity[i,j]=CalculateDist(user1,user2,'cosine')
    return Similarity

# find the index of the neighbor users
def NeighborUser(Similarity,TopK):
    Similarity=Similarity-identity(Similarity.shape[0])
    ind=argsort(Similarity)
    NeighborInd=ind[:,-TopK:]
    return NeighborInd


def recomendTopN(NeighborData,topN):  #give the predicted score as the weighted sum of its neighbor's score
    sign=NeighborData>0
    for i in range(sign.shape[1]):
        ind=where(sign[:,i]==1)
        ind=ind[0]
        if len(ind)>topN:
            NeighborData[:ind[-topN],i]=0
    num=sum(sign,0)
    num[num==0]=1
    num[num>topN]=topN
    recommend=sum(NeighborData,0)/num
    return recommend


def Fill(Data,Neighbor):     #Filled the user-item score matrix accroding to the neighbor users' score
    DataFill=Data
    for i in range(Data.shape[0]):
        user = Data[i, :]
        ind = where(user > 0)
        ind = ind[0]
        NeighborData=Data[Neighbor[i,:],:]
        recommend=recomendTopN(NeighborData,50)
        recommend[ind]=user[ind]
        DataFill[i,:]=recommend
    DataFill=round(DataFill)
    DataFill[DataFill==0]=4
    return DataFill



def Ubcf(Data):    #user based collaborative filter
    Similarity=SimilarityMat(Data)
    Neighbor=NeighborUser(Similarity,800)
    DataFill=Fill(Data,Neighbor)
    return DataFill


def Ibcf(Data):    #Item based collaborative filter, which can be easily achieved by transforming the user-item score matrix
    Data=Data.T
    DataFill=Ubcf(Data)
    DataFill=DataFill.T
    return DataFill


def evaluate(DataFill,TrueData):    #evaluate the MAE
    predict=DataFill.flatten()
    label=TrueData.flatten()
    ind=where(label>0);ind=ind[0]
    error=sum(abs(label[ind]-predict[ind]))
    print 'MAE= '
    print error/20000



def output(DataFill):   #convert the score matrix to needed format
    DataFill=DataFill.flatten()
    Data=zeros((len(DataFill),3))
    Data[:,2]=DataFill
    for i in range(len(DataFill)):
        Data[i,0]=i/1682+1
        Data[i,1]=i%1682+1
    savetxt('Data.txt',Data,fmt='%d',delimiter=' ',newline='\n')
    return Data



def main():
    #DataPrepare()
    Data=load('Data.npy')
    TrueData=load('TrueData.npy')
    DataFill=Ubcf(Data)
    save('DataFill.npy',DataFill)
    evaluate(DataFill,TrueData)
    Datatxt=output(DataFill)


if __name__=='__main__':
    main()
