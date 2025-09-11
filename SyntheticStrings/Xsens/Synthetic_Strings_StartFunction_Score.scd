//Start the function

~mocap_Soni_Strings_A_1_xsens.value

//CONFI 1


//originales
~joints=      [10,9,8,5,6,17];
~jointsRef= [14,13,12,21,22,18];
//cambiar joints. Arms to opposite legs
~joints=      [10,9,8,14,13,12];
~jointsRef= [22,21,20,18,17,16];

//Change joints. Arms to spine
~joints= [10,9,8,14,13,12];
~jointsRef= [0,1,2,3,4,5,6];

//Change joints. hands arms to other part including shoulder + head and feet
~joints=       [10,14,8,9,5,6];
~jointsRef=  [11,7,13,12,21,22];

//Change hands to many joints
~joints=      [10,14,10,14, 10,14];
~jointsRef= [0,1,2,3,4,6];

~jerkGlobMeanLength= 1; ~accGlobMeanLength=2;
~accNorMaxValue= 15;
~jerkNorMaxValue=35;
~stopValue= 1;
~threshold=0.025;

~maxAmp= 12     !     ~jointSelectDict.size;
~minAmp=0.05  ! ~jointSelectDict.size;
~ampCurve= 0 ! ~jointSelectDict.size;

~minFreqRate= 7 ! ~jointSelectDict.size;
~maxFreqRate= 30 ! ~jointSelectDict.size;
~freqRateCurve= -3 ! ~jointSelectDict.size;

~minFreqMul=10 ! ~jointSelectDict.size;
~maxFreqMul=34.midicps ! ~jointSelectDict.size;
~freqMulCurve= -2 ! ~jointSelectDict.size;

~minHarmAmp=1.5; ~maxHarmAmp= 0.5; ~armAmpCurve= -1;

~minCombDecay=2.25 ! ~jointSelectDict.size;
~maxCombDecay=0.5! ~jointSelectDict.size;
~combDecayCurve=0    ! ~jointSelectDict.size;

//pitch
~minPitch=  36.midicps ! ~voices;
~maxPitch= 72.midicps ! ~voices;


//reverb
~revDryWet= 0.75; ~revLowFreq= 100;  ~revMidFreq= 6000; ~revTimeLow= 3; ~revTimeMid= 3.75;  ~revPreDelay= 0.1;


//select some voices only

//Change joints. hands arms to arms
~joints= [20,18,17,10,8,6];
~jointsRef= [10,8,6,20,18,16];

//one voice between hands
~maxAmp= [10,0,0,0,0,0];
~minAmp=[0.2,0.0,0,0,0,0];
~ampCurve= 0 ! ~jointSelectDict.size;

//cambiar joints. Arms to opposite legs
~joints= [20,18,17,10,8,6];
~jointsRef= [28,27,26,32,31,30];

//very high
~minPitch=  12.midicps ! ~voices;
~maxPitch= 84.midicps ! ~voices;

//very low
~minPitch=  12.midicps ! ~voices;
~maxPitch= 24.midicps ! ~voices;

//mid
~minPitch=  24.midicps ! ~voices;
~maxPitch= 60.midicps ! ~voices;

//mid High
~minPitch=  60.midicps ! ~voices;
~maxPitch= 72.midicps ! ~voices;