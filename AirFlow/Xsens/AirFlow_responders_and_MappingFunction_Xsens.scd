
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
};

~eM_mocap_Soni_flute_A_1={
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
}