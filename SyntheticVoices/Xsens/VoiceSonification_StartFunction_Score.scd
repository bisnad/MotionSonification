//Start the function

~mocap_Soni_fof_A_2_xsens_V3.value;

//parameters
//que suene de fondo
~accNorMaxValue=100;
~rotAccNorMaxValue=1000;
~jerkNorMaxValue=250;
~threshold=0.005;
~stopValue=1;
~accGlobMeanLength= 2;
~jerkGlobMeanLength= 1;
~minAmp=0.00 ! ~jointsSelection.size;
~maxAmp=1! ~jointsSelection.size;
~ampCurve= 0 ! ~jointsSelection.size;

//Listern right hand only
~maxAmp=0! ~jointsSelection.size;   ~maxAmp.put(~jointSelectDict.rightHand,1); //

~minLpf= 2000 ! ~jointsSelection.size;
~maxLpf=22000! ~jointsSelection.size;
~lpfCurve= -2 ! ~jointsSelection.size;
~minPitch=12.midicps ! ~jointsSelection.size;
~maxPitch=86.midicps! ~jointsSelection.size;

~minVibDepth=0.0 ! ~jointsSelection.size;
~maxVibDepth=2.5 ! ~jointsSelection.size;
~vibDepthCurve= -1.5 ! ~jointsSelection.size;

~revTime=1.50; ~damping= 0.6; ~dryLevel= -6;

//Other tessituras
~minPitch=24.midicps ! ~jointsSelection.size;
~maxPitch=48.midicps ! ~jointsSelection.size;


//Other tessituras
~minPitch=48.midicps ! ~jointsSelection.size;
~maxPitch=52.midicps ! ~jointsSelection.size;

//Experiment with pitch ranges
~minPitch=65.midicps ! ~jointsSelection.size;
~maxPitch=72.midicps ! ~jointsSelection.size;


//Some joints
~maxAmp=0! ~jointsSelection.size;   ~maxAmp.put(~jointSelectDict.rightHand,1); //escucha solo mano derech
~maxAmp.put(~jointSelectDict.rightForeArm,1); //escucha también el antrebrazo

~maxAmp.put(~jointSelectDict.leftHand,1); //escucha también mano izquierda
~maxAmp.put(~jointSelectDict.leftForeArm,1); //escucha también mano izquierda


//turn all joints down
(0..22).collect{|x| ~maxAmp [x]= 0};
(0..22).collect{|x| ~minAmp [x]= 0};
//Listen feet only for example
[17,21].collect{|x| ~maxAmp [x]= 2};
[17,21].collect{|x| ~minAmp [x]= 0};

//or written like this. Listen right hand only
~maxAmp=0!~jointsSelection.size;   ~maxAmp.put(~jointSelectDict.rightHand,1);
