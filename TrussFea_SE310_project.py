import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Problem Definition


file=pd.ExcelFile('InputFea_a.xlsx')
elems=file.parse(0)
nodes=file.parse(1)
CSinfo=file.parse(2)
Boundary=file.parse(3)
Forcing=file.parse(4)
nodeCords=nodes.values
elemNodes=elems.values
Area=CSinfo.values[:,0]
modE=CSinfo.values[:,1]
DispCon=Boundary.values
Fval=Forcing.values
scale=1


#Problem Initialization
nELEM=elemNodes.shape[0]
nNODE=nodeCords.shape[0]
nDC=DispCon.shape[0]
nFval=Fval.shape[0]
NDOF=nNODE*2
uDisp=np.zeros((NDOF,1))
forces=np.zeros((NDOF,1))
Stiffness=np.zeros((NDOF,NDOF))
Stress=np.zeros((nELEM))
kdof=np.zeros((nDC))
xx=nodeCords[:,1]
yy=nodeCords[:,2]
L_elem=np.zeros((nELEM))


#Building the displacement array
for i in range(nDC):
    indice=DispCon[i,:]
    v=indice[2]
    v=v.astype(float)
    indice=indice.astype(int)
    kdof[i]=indice[0]*2+indice[1]-1
    uDisp[indice[0]*2+indice[1]-1]=v
    
#Building the force array

for i in range(nFval):
    indice2=Fval[i,:]
    v=indice2[2];
    v=v.astype(float)
    indice2=indice2.astype(int)
    forces[indice2[0]*2+indice2[1]-1]=v



#Identifying known and unknown displacement degree of freedom
kdof=kdof.astype(int)
ukdof=np.setdiff1d(np.arange(NDOF),kdof)


#Loop over all the elements

for e in range(nELEM):
    indiceE=elemNodes[e,:]
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1])
    elemDOF=elemDOF.astype(int)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya)
    c=xa/len_elem
    s=ya/len_elem


    #update truss forces
    vol=len_elem*Ae
    Fg=vol*8050*9.81*0
    forces[indiceE[0]*2+1]+=-Fg/2
    forces[indiceE[1]*2+1]+=-Fg/2

    
    # Step 1. Define elemental stiffness matrix
    ke=(Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])
    
    
    # Step 2. Transform elemental stiffness matrix from local to global coordinate system
    T=np.array([[c,s,0,0],[0,0,c,s]])
    k2=np.matmul(T.transpose(),np.matmul(ke,T))

    
    # Step 3. Assemble elemental stiffness matrices into a global stiffness matrix
    Stiffness[np.ix_(elemDOF,elemDOF)] +=k2
    
# Step 4. Partition the stiffness matrix into known and unknown dofs
k11=Stiffness[np.ix_(ukdof,ukdof)]
k12=Stiffness[np.ix_(ukdof,kdof)]
k21=k12.transpose()
k22=Stiffness[np.ix_(kdof,kdof)]




# Step 4a. Solve for the unknown dofs and reaction forces
f_known=forces[ukdof]-np.matmul(k12,uDisp[kdof])
uDisp[np.ix_(ukdof)]=np.linalg.solve(k11,f_known)

forces[np.ix_(kdof)]=np.matmul(k21,uDisp[np.ix_(ukdof)])+np.matmul(k22,uDisp[np.ix_(kdof)])

plt.figure(300)

# Step 5. Evaluating Internal Forces and stresses
for e in range(nELEM):
    indiceE=elemNodes[e,:]
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1])
    elemDOF=elemDOF.astype(int)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya)
    L_elem[e]=len_elem
    c=xa/len_elem
    s=ya/len_elem
    
    
    ke=(Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])
    
    T=np.array([[c,s,0.0,0.0],[0.0,0.0,c,s]])
    
    Fint=np.matmul(ke,np.matmul(T,uDisp[np.ix_(elemDOF)]))
    Stress[e]=Fint[1]/Ae
    plt.plot(np.array([xx[indiceE[0]],xx[indiceE[1]]]),np.array([yy[indiceE[0]],yy[indiceE[1]]]))
    plt.plot(np.array([xx[indiceE[0]]+uDisp[indiceE[0]*2]*scale,xx[indiceE[1]]+uDisp[indiceE[1]*2]*scale]),np.array([yy[indiceE[0]]+uDisp[indiceE[0]*2+1]*scale,yy[indiceE[1]]+uDisp[indiceE[1]*2+1]*scale]),'--')


plt.xlim(min(xx)-abs(max(xx)/10), max(xx)+abs(max(xx)/10))
plt.ylim(min(yy)-abs(max(yy)/10), max(yy)+abs(max(xx)/10))
plt.gca().set_aspect('equal', adjustable='box')
pduDisp = pd.DataFrame({'disp': uDisp[:,0]})
pdforces=pd.DataFrame({'forces': forces[:,0]})
pdStress=pd.DataFrame({'Stress': Stress})
pdLen=pd.DataFrame({'Length': L_elem})
#Displaying the results
print(pduDisp)
print(pdforces)
print(pdStress)
pduDisp.to_excel("uDisp.xlsx",sheet_name='uDisp')  
pdforces.to_excel("forces.xlsx",sheet_name='forces')  
pdStress.to_excel("Stress.xlsx",sheet_name='Stress')  
pdLen.to_excel("Length.xlsx",sheet_name='Length')  

plt.show()

