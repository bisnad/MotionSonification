

//con la flauta antigua y con la neuva acelaración y nuevo jerk


//a single responder for several types of data

(
~mocapResp_RotPos={ ['/mocap/joint/pos_world', '/mocap/joint/rot','/mocap/joint/acceleration','/mocap/joint/rot_acceleration','/mocap/joint/lin_acc','/mocap/joint/rot_acc', '/mocap/joint/ang_jerk','/mocap/joint/lin_jerk'].collect{|data| OSCFunc({arg msg, time, addr;
				var parameter, value;


				parameter      =		 msg[0];
				value          =		 msg[1..];

	//msg.postln;
	//value[x].postln;

	~functionMocap_data_1[parameter.asSymbol].value(msg[1..])

}, data, nil, 9005 ;
)
}
}
)

//EM_mocap_Soni_flute_A_1
//flauta original
//con MUR_StacatoMicroMicroSlow_01_mb_proc
//con MUR_ContinuityInterrupted_01_mb_proc
//con MUR_Rigidity_02_mb_proc pero freq con rotation mejor
//ojo , quizá es mejor con jerk y no rot jerk
(
var synths,reverb,responders, vowels, joint= (0*3), senseWorldAcc, senseWorldjerk, accNorMaxValue=2,accValues=(0..2)!voices,accRotValues= (0..2)!voices,
jointsSelection= ~jointsSelection, voices=jointsSelection.size, accValuesGlobal=0!jointsSelection.size ;


//senseWorldAcc= SensorData.new !77; //así no funciona la iteración si itero de nuevo
senseWorldAcc=voices.collect{SensorData.new }; //así si
senseWorldjerk=voices.collect{SensorData.new }; //así si


~mocapResp_RotPos.value;
~accNorMaxValue=50;
~rotAccNorMaxValue=300;
~jerkNorMaxValue=300;
~jerkGlobMeanLength= 1; ~accGlobMeanLength= 2;
~minPitch=  24.midicps ! jointsSelection.size;
~maxPitch= 73.midicps ! jointsSelection.size;
~maxAmp=1! jointsSelection.size;
~minAmp=0.0002! jointsSelection.size;
~ampCurve=1! jointsSelection.size;
~maxPan= 2;
~stopValue= 1;
~threshold=0.05;
/*~minPitch=[48,48,24,24].midicps;
~maxPitch=[76,76,84,84].midicps;*/
~fbk1min=0.28 ! jointsSelection.size;
~fbk1max=0.51! jointsSelection.size;
~fbk2min=0.2   ! jointsSelection.size;
~fbk2max=0.35! jointsSelection.size;
~fbkCurve= 0    ! jointsSelection.size;
~minBreath=0.005 ! jointsSelection.size;
~maxBreath=0.05! jointsSelection.size;
~breathCurve=0    ! jointsSelection.size;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.0; ~flowFbkFactor=1 ; ~enFbkFactor=0;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.1;
~minLpf=2000; ~maxLpf=5000; ~lpfCurve=0;
~flowBreathFactor= 1; ~enBreathFactor=0;
~minVibDepth=0;~maxVibDepth=1; ~vibDepthCurve=1; ~maxVibRate=8; ~minVibRate=8;

~accNorMaxValue= 12; ~jerkNorMaxValue= 35;
synths= voices.collect{|x|Synth(\waveguideFluteMono_lag,[
         \freq,  60, \lagFreq, 0, \vibWidth,0.0, \vibRate,6,
	\amp, 0.2, \amp2, 0,  \lagAmp, 0.5,\attack, 0.1, \attackNoise, 0.9,    \legato,1.0,\relNoise,2.25, \rel,2.25,
		\ipress,  0.9 ,  \lpf, 2000, \ibreath, 0.03, \lagBreath, 0, \ifeedbk1, 0.4, \ifeedbk2, 0.4,
	\reverbMix, 1,\roomSize, 0.5,  \reverbDamp, 0.25,
	\out, (30 +(x%8))
	]);
};

reverb=8.collect{|x|Synth( \itin_Verb3mono, [\out, 0 +x, \in,30 +x ,\taillevel,-12, \revtime,1.75, \roomsize,143, \damping,0.09, \spread,30.0, \drylevel, -2,  \addAction, 1, \dur, 1], addAction: \addToTail)};

~functionMocap_data_1= (
	'/mocap/joint/pos_world':     {| valor | var  posX=valor[(0,3..valor.size-1)], posY=valor[(1,4..valor.size)], posZ=valor[(joint+2, (joint+5)..valor.size)],minPosZ_Value= 0, maxPosZ_Value= 3,minPosY_Value= -5, maxPosY_Value=5;
		posZ =posZ[jointsSelection];
		posY =posY[jointsSelection];
		(voices).do{|x| synths[x].set(
			\freq, posZ[x].linlin(minPosZ_Value,maxPosZ_Value,~minPitch[x],~maxPitch[x])* (x.linlin(0,voices,1,1.1)),
			 //\panPos, posY[x].linlin(minPosY_Value,1.0,maxPosY_Value,2.0),
			)
		};
		//pan[0].postln
	},


		'/mocap/joint/rot':     {| valor | var  rotX=valor[(joint+0,(joint+3)..valor.size-1)], rotY=valor[(joint+1,(joint+4)..valor.size)], rotZ=valor[(joint+2, (joint+5)..valor.size)],minRotX_Value= -pi, maxRotX_Value= pi,minRotY_Value= -pi/2, maxRotY_Value= pi/2,minRotZ_Value= -pi, maxRotZ_Value= pi;

		rotX= rotX[jointsSelection];
	    rotY= rotY[jointsSelection];
	    rotZ= rotZ[jointsSelection];
/*vowelSpaceWhich=rotX;
vowelSpacePitch=rotY;*/



		(voices).do{|x| synths[x].set(
			//\freq, rotY[x].linlin(minRotY_Value,maxRotY_Value,~minPitch[x],~maxPitch[x])* (x.linlin(0,voices,1,1.1)),
			//\freq, rotX[x].linlin(minRotX_Value,maxRotX_Value,~minPitch,~maxPitch)* (x.linlin(0,voices,1,1.1)),
		\panPos, rotZ[x].linlin(-pi,pi,0.0,2.0),
			)
		};

	},
		'/mocap/joint/rot_acc':     {| valor |var values, jointsGrouped, rotAccNorMaxValue= 1000;
		rotAccNorMaxValue= ~rotAccNorMaxValue;
		jointsGrouped=(0,3..valor.size-1).collect{|limit,index| valor[limit..(limit +2)]};
		jointsGrouped=jointsGrouped.collect{|x|x.linlin(0,rotAccNorMaxValue,0,1)}; //lo normalizo
        accRotValues=jointsGrouped;
		//jointsGrouped[10].postln;
		                                           },
		'/mocap/joint/lin_acc':     {| valor |var values, means, jointsGrouped,accNorMaxValue=100,globalStillness;
	//means=(0,3..valor.size-1).collect{|limit,index| valor[limit..(limit +2)].abs.mean}; //encuentro las medias de cada articulación.

accNorMaxValue= ~accNorMaxValue;
		//esta sección para hacer la media con la  acceleración rotacional
		jointsGrouped=(0,3..valor.size-1).collect{|limit,index| valor[limit..(limit +2)]};
		jointsGrouped=jointsGrouped.collect{|x|x.linlin(0,accNorMaxValue,0,1)}; //lo normalizo
		 accValues=jointsGrouped;
		means= voices.collect{|x| (accRotValues[x]++accValues[x]).abs.maxItem}; //los uno y saco la media o maxitem

     	//means=means.collect{|x|x.linlin(0,accNorMaxValue,0,1)}; //lo normalizo


	   means= means[jointsSelection ];

		means.size.collect{|x|senseWorldAcc[x].addValue(means[x])};
		means.size.collect{|x|senseWorldAcc[x].stlen_( ~accGlobMeanLength )};
		means.size.collect{|x|means[x]=senseWorldAcc[x].shortMean} ;
	globalStillness=means.size.collect{|x|if(means[x]<~threshold) {~stopValue}{1}};
		accValuesGlobal=means; //para sacarlos y utilizarlo en el jerk
		//~viewData=	means[41];
		//means[41].postln;

		(voices).do{|x| synths[x].set(
			\amp2, ControlSpec(~minAmp[x],~maxAmp[x], ~ampCurve[x]).map((means[x]))*globalStillness[x] ,
			\lpf, ControlSpec(~minLpf,~maxLpf, ~lpfCurve).map((means[x])) ,
                                                  )
		                 }
		                                               },
			'/mocap/joint/ang_jerk':     {| valor |var values, means, jerkNorMaxValue=500;
		//means=(0,3..valor.size-1).collect{|limit,index| valor[limit..(limit +2)].abs.mean};
	//~viewData=	means[11];

       jerkNorMaxValue= ~jerkNorMaxValue;
      means=valor.collect{|x|x.linlin(0,jerkNorMaxValue,0,1)}; //lo normalizo
	    means= means[jointsSelection ];
		means.size.collect{|x|senseWorldjerk[x].addValue(means[x])};
		means.size.collect{|x|senseWorldjerk[x].stlen_( ~jerkGlobMeanLength )};
		means.size.collect{|x|means[x]=senseWorldjerk[x].shortMean} ;
		//~viewData=	means[11];

				(voices).do{|x| synths[x].set(
			\ifeedbk1, ((ControlSpec(~fbk1min[x],~fbk1max[x],~fbkCurve[x]).map(means[x])*~flowFbkFactor) +
				(ControlSpec(~fbk1min[x],~fbk1max[x],~fbkCurve[x]).map(accValuesGlobal[x])*~enFbkFactor)) ,
			\ifeedbk2,((ControlSpec(~fbk2min[x],~fbk2max[x],~fbkCurve[x]).map(means[x])*~flowFbkFactor) +
				(ControlSpec(~fbk2min[x],~fbk2max[x],~fbkCurve[x]).map(accValuesGlobal[x])*~enFbkFactor)) ,
			\ibreath, ((ControlSpec(~minBreath[x],~maxBreath[x],~breathCurve[x]).map(means[x])*~flowBreathFactor)+
				(ControlSpec(~minBreath[x],~maxBreath[x],~breathCurve[x]).map(accValuesGlobal[x])*~enBreathFactor)),
		\vibDepth, ControlSpec(~minVibDepth,~maxVibDepth, ~vibDepthCurve).map((means[x])).midiratio
                                                  )
		                 }

		                                                 },

);
~jointSelectDict.size=  jointsSelection.size
)



//CONFI 1
//separar el acceso por articulación también para el ffeedback
//Drawing with air
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

//cambiar el registro de todas las articulaciones. todas en un mismo rango
~minPitch=  48.midicps ! ~jointSelectDict.size;
~maxPitch= 60.midicps ! ~jointSelectDict.size;



//CONFI 2
//DEL DAFNE
//MUR_ContinuityDifferentQualities_01_mb_proc
//cambiar el registro de todas las articulaciones. todas en un mismo rango
~stopValue= 1;
~threshold=0.01;
~jerkGlobMeanLength= 3; ~accGlobMeanLength= 2;
~minPitch=  58.midicps ! ~jointSelectDict.size;
~maxPitch= 64.midicps ! ~jointSelectDict.size;
~maxAmp= 0.5         !     ~jointSelectDict.size;
~minAmp=0.05  ! ~jointSelectDict.size;
~ampCurve= 0 ! ~jointSelectDict.size;
~fbk1min=0.35 ! ~jointSelectDict.size;
~fbk1max=0.56! ~jointSelectDict.size;
~fbk2min=0.32   ! ~jointSelectDict.size;
~fbk2max=0.53 ! ~jointSelectDict.size;
~fbkCurve= 0    ! ~jointSelectDict.size;
~flowFbkFactor=1 ; ~enFbkFactor=0;
~minBreath=0.005 ! ~jointSelectDict.size;
~maxBreath=0.075! ~jointSelectDict.size;

//MEDIO SENSIBLE
~jerkGlobMeanLength= 6; ~accGlobMeanLength= 6;

//MENOS SENSIBLE
~jerkGlobMeanLength= 13; ~accGlobMeanLength= 12;


//MUY SENSIBLE
~jerkGlobMeanLength= 2; ~accGlobMeanLength= 2;

~minPitch=  24.midicps ! ~jointSelectDict.size;
~maxPitch= 35.midicps ! ~jointSelectDict.size;

//CONFI 2. 1
//apolo

//CONFI 3
~stopValue= 0;
~threshold=0.05;
~jerkGlobMeanLength= 5; ~accGlobMeanLength= 8;
//Diferentes tessituras para parte superior e inferir del cuerpo
//ojo
//cambiar el registro de algunas articulaciones. Los pies más graves
[~jointSelectDict.leftFoot, ~jointSelectDict.rightFoot].collect{|x| ~minPitch [x]= 36.midicps; ~maxPitch [x]= 48.midicps};  //36 o 24
[~jointSelectDict.leftToe, ~jointSelectDict.rightToe].collect{|x| ~minPitch [x]= 35.midicps; ~maxPitch [x]= 47.midicps};  //36 o 24

//o parte inferior más grave
[~jointSelectDict.leftFoot, ~jointSelectDict.rightFoot].collect{|x| ~minPitch [x]= 36.midicps; ~maxPitch [x]= 48.midicps};  //36 o 24
[~jointSelectDict.leftToe, ~jointSelectDict.rightToe].collect{|x| ~minPitch [x]= 35.midicps; ~maxPitch [x]= 47.midicps};  //36 o 24


//cambiar el registro de algunas articulaciones. el resto más agudo
(0..25).collect{|x| ~minPitch [x]= 70.midicps};
(0..25).collect{|x| ~maxPitch [x]= 75.midicps};

//distinguir acentuar el feedback de una o varias articulaciones. Piernas y pies
(26..33).collect{|x| ~fbk1min [x]= 0.28};
(26..33).collect{|x| ~fbk1max [x]= 0.51};
(26..33).collect{|x| ~fbk2min [x]= 0.28};
(26..33).collect{|x| ~fbk2max [x]= 0.537};

//distinguir el feedback de una o varias articulaciones. torso y brazos
(0..25).collect{|x| ~fbk1min [x]= 0.28};
(0..25).collect{|x| ~fbk1max [x]= 0.51};
(0..25).collect{|x| ~fbk2min [x]= 0.26};
(0..25).collect{|x| ~fbk2max [x]= 0.35};

//aumentando el breath solo en las piernas
(26..33).collect{|x| ~minBreath [x]= 0.005};
(26..33).collect{|x| ~maxBreath [x]= 0.05};
//aumentando el breath lo disminuyo en brazos
(0..25).collect{|x| ~minBreath [x]= 0.005};
(0..25).collect{|x| ~maxBreath [x]= 0.025};

//CONFI 3.1
//cada parte corporal en un registro
~stopValue= 0;
~threshold=0.025;
~jerkGlobMeanLength= 2; ~accGlobMeanLength= 5;
~maxAmp=0.7         !     ~jointSelectDict.size;
~minAmp=0.0002  ! ~jointSelectDict.size;
//Diferentes tessituras para parte superior e inferir del cuerpo

//head
~maxAmp[4,5]=1;
~minAmp=0.0002  ! ~jointSelectDict.size;

//registro cabeza
(4..5).collect{|x| ~minPitch [x]= 70.midicps};
(4..5).collect{|x| ~maxPitch [x]= 79.midicps};

//feedback cabeza
(4..5).collect{|x| ~fbk1min [x]= 0.25};
(4..5).collect{|x| ~fbk1max [x]= 0.55};
(4..5).collect{|x| ~fbk2min [x]= 0.26};
(4..5).collect{|x| ~fbk2max [x]= 0.55};

//breath cabeza
(4..5).collect{|x| ~minBreath [x]= 0.005};
(4..5).collect{|x| ~maxBreath [x]= 0.05};

//amp cabeza
(4..5).collect{|x| ~maxAmp [x]= 0.7};
(4..5).collect{|x| ~minAmp [x]= 0.0002.midicps};

//registro torso
(0..3).collect{|x| ~minPitch [x]= 65.midicps};
(0..3).collect{|x| ~maxPitch [x]= 70.midicps};

//feedback torso
(0..3).collect{|x| ~fbk1min [x]= 0.3};
(0..3).collect{|x| ~fbk1max [x]= 0.51};
(0..3).collect{|x| ~fbk2min [x]= 0.35};
(0..3).collect{|x| ~fbk2max [x]= 0.4};

//breath torso
(0..3).collect{|x| ~minBreath [x]= 0.005};
(0..3).collect{|x| ~maxBreath [x]= 0.052};


//registro hombros
[6,16].collect{|x| ~minPitch [x]= 55.midicps};
[6,16].collect{|x| ~maxPitch [x]= 60.midicps};

//feedback hombros
[6,16].collect{|x| ~fbk1min [x]= 0.35};
[6,16].collect{|x| ~fbk1max [x]= 0.51};
[6,16].collect{|x| ~fbk2min [x]= 0.34};
[6,16].collect{|x| ~fbk2max [x]= 0.45};

//breath hombros
[6,16].collect{|x| ~minBreath [x]= 0.0095};
[6,16].collect{|x| ~maxBreath [x]= 0.05};


//registro brazos
[7,8,9,17,18,19].collect{|x| ~minPitch [x]= 45.midicps};
[7,8,9,17,18,19].collect{|x| ~maxPitch [x]= 50.midicps};

//feedback brazos
[7,8,9,17,18,19].collect{|x| ~fbk1min [x]= 0.35};
[7,8,9,17,18,19].collect{|x| ~fbk1max [x]= 0.51};
[7,8,9,17,18,19].collect{|x| ~fbk2min [x]= 0.34};
[7,8,9,17,18,19].collect{|x| ~fbk2max [x]= 0.4};

//breath brazos
[7,8,9,17,18,19].collect{|x| ~minBreath [x]= 0.005};
[7,8,9,17,18,19].collect{|x| ~maxBreath [x]= 0.035};


//registro manos
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~minPitch [x]= 50.midicps};
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~maxPitch [x]= 55.midicps};

//feedback manos
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~fbk1min [x]= 0.35};
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~fbk1max [x]= 0.51};
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~fbk2min [x]= 0.34};
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~fbk2max [x]= 0.45};

//breath manos
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~minBreath [x]= 0.005};
[10,11,12,13,14,15,20,21,22,23,24,25].collect{|x| ~maxBreath [x]= 0.03};



//registro Piernas
[26,27,30,31].collect{|x| ~minPitch [x]= 40.midicps};
[26,27,30,31].collect{|x| ~maxPitch [x]= 45.midicps};

//feedback Piernas
[26,27,30,31].collect{|x| ~fbk1min [x]= 0.3};
[26,27,30,31].collect{|x| ~fbk1max [x]= 0.51};
[26,27,30,31].collect{|x| ~fbk2min [x]= 0.34};
[26,27,30,31].collect{|x| ~fbk2max [x]= 0.45};

//breath piernas
[26,27,30,31].collect{|x| ~minBreath [x]= 0.005};
[26,27,30,31].collect{|x| ~maxBreath [x]= 0.035};


//registro Pies
[28,29,32,33].collect{|x| ~minPitch [x]= 24.midicps};
[28,29,32,33].collect{|x| ~maxPitch [x]= 36.midicps};

//feedback Pies
[28,29,32,33].collect{|x| ~fbk1min [x]= 0.35};
[28,29,32,33].collect{|x| ~fbk1max [x]= 0.51};
[28,29,32,33].collect{|x| ~fbk2min [x]= 0.34};
[28,29,32,33].collect{|x| ~fbk2max [x]= 0.55};

//breath Pies
[28,29,32,33].collect{|x| ~minBreath [x]= 0.005};
[28,29,32,33].collect{|x| ~maxBreath [x]= 0.075};









