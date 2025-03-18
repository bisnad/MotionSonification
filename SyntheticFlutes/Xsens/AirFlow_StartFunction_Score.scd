//encender nueva funcion
~rO_flute_interact_B_1.value;


//paramters
~accNorMaxValue=50;
~rotAccNorMaxValue=200;
~jerkNorMaxValue=500;
~stopValue= 1;
~threshold=0.025;
~minPitch=  24.midicps ! ~jointSelectDict.size;
~maxPitch= 73.midicps ! ~jointSelectDict.size;
~jerkGlobMeanLength= 1; ~accGlobMeanLength= 3;
~maxAmp=1         !     ~jointSelectDict.size;
~minAmp=0.0002  ! ~jointSelectDict.size;
~ampCurve= -1! ~jointSelectDict.size;
~fbk1min=0.28 ! ~jointSelectDict.size;
~fbk1max=0.51! ~jointSelectDict.size;
~fbk2min=0.2   ! ~jointSelectDict.size;
~fbk2max=0.45 ! ~jointSelectDict.size;
~fbkCurve= -2    ! ~jointSelectDict.size;
~flowFbkFactor= 0.8 ; ~enFbkFactor=0.2;
~flowBreathFactor= 0.5; ~enBreathFactor=0.5;
~minLpf=2000; ~maxLpf=5000; ~lpfCurve=0;
~minBreath=0.005 ! ~jointSelectDict.size;
~maxBreath=0.075! ~jointSelectDict.size;
~breathCurve=0    ! ~jointSelectDict.size;
~minVibDepth=0;~maxVibDepth=0; ~vibDepthCurve=1; ~maxVibRate=8; ~minVibRate=8;

//change all joints
~minPitch=  48.midicps ! ~jointSelectDict.size;
~maxPitch= 60.midicps ! ~jointSelectDict.size;


//turn all joints down
(0..22).collect{|x| ~maxAmp [x]= 0};
(0..22).collect{|x| ~minAmp [x]= 0};
//Listen feet only for example
[17,21].collect{|x| ~maxAmp [x]= 2};
[17,21].collect{|x| ~minAmp [x]= 0};

//or written like this. Listen right hand only

~maxAmp=0!~jointsSelection.size;   ~maxAmp.put(~jointSelectDict.rightHand,1); //escuchar solo la mano derecha
