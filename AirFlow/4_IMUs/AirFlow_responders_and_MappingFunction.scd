~responders_absIMU_raw={~respondersRaw=['/imu/1', '/imu/2','/imu/3','/imu/4'
].collect{|imu, index|OSCFunc({arg msg, time, addr;
				var parameter, value;


				parameter      =		 msg[0];
				value          =		 msg[0..];

		//msg.postln;


	~function[parameter.asSymbol].value(value)

}, imu, nil, 10001 +index;
)
}
};


~respondersJerk2={~respondersFlow2=['/flow/1','/flow/2','/flow/3','/flow/4'
].collect{|feature|OSCFunc({arg message, time, addr;
				var parameter, value;


				parameter      =		 message[0];
				value          =		 message[0..];

	//message.postln;


	~functionFlow[parameter.asSymbol].value(message[1])

}, feature, nil, 57120;
)
}
};


//this is just to forward the imu data jerk to  the jerk app
~forwardJerk={
['/imu/1','/imu/2','/imu/3','/imu/4'].collect{|imu, index|
var addr = NetAddr.new("127.0.0.1", 13001);
OSCdef(imu, {
  arg msg;
	//msg.postln;
  addr.sendMsg(*msg);
}, imu, nil, 10001 +index);
}};



//RO_flute_interact_B_1
//model for any imu number
~rO_flute_interact_B_1={
var synths, reverb,responder, numVoices=4, lagAmp=0.1, basefreq=50, mulFreq=0,
lag=0.5, senseTheWorld,senseWorldFlow, flowNorMaxValue=1200, imu_EnVal= 0!4, imu_FlowVal= 0!4,  freq=0!4, axisY=0!4, freqFbkFactor= 1!4 ;



~maxAmp=1; ~minAmp=0.0002; ~ampCurve=1;
~maxPan= 2;
~stopValue= 1;
~threshold=0.05;
~minPitch=[48,48,24,24].midicps;
~maxPitch=[76,76,84,84].midicps;
~fbk1min=0.28; ~fbk1max=0.65; ~fbk2min=0.2; ~fbk2max=0.3; ~fbkCurve= 0;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.0; ~flowFbkFactor=1 ; ~enFbkFactor=0;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.1;
~minLpf=2000; ~maxLpf=5000;
~minBreath=0.005; ~maxBreath=0.1; ~breathCurve=0; ~flowBreathFactor= 1; ~enBreathFactor=0;

~maxVibDepth= 3.0; ~maxVibRate=8; ~minVibRate=8;
~dataMeanLengthFlow=6; ~dataMeanLengthEnergy=3;

senseTheWorld= [ SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new];

senseWorldFlow=[ SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new,SensorData.new];

~responders_absIMU_raw.value;
~respondersJerk2.value;
~forwardJerk.value;
~synths= numVoices.collect{|x|Synth(\waveguideFluteMono_lag,[
         \freq,  60, \lagFreq, 0, \vibWidth,0.0, \vibRate,6,
	\amp, 0.2, \amp2, 0,  \lagAmp, 0.1,\attack, 0.1, \attackNoise, 0.9,    \legato,1.0,\relNoise,2.25, \rel,2.25,
		\ipress,  0.9 ,  \lpf, 2000, \ibreath, 0.03, \lagBreath, 0, \ifeedbk1, 0.4, \ifeedbk2, 0.4,
	\reverbMix, 1,\roomSize, 0.5,  \reverbDamp, 0.25,
	\out, 30 +x
	]);
};

	~reverb=4.collect{|x|Synth( \itin_Verb3mono, [\out, 0 +x, \in,30 +x ,\taillevel,-12, \revtime,2.75, \roomsize,143, \damping,0.3, \spread,30.0, \drylevel, -2,  \addAction, 1, \dur, 1], addAction: \addToTail)};

~function= ();
~functionFlow= ();

[ '/flow/1','/flow/2','/flow/3','/flow/4'].do{ |imu, index|  ~functionFlow.add(imu.asSymbol->   {| valor |

	var specBreath= ControlSpec(~minBreath,~maxBreath,~breathCurve), specFbk1= ControlSpec(~fbk1min,~fbk1max,~fbkCurve),
	specFbk2= ControlSpec(~fbk2min,~fbk2max,~fbkCurve), specVibRate=ControlSpec(~minVibRate,~maxVibRate,0), specVibDepth= ControlSpec(0.0,~maxVibDepth,0), relevantValue;

	imu_FlowVal[index]=valor.linlin(0,flowNorMaxValue,0,1);
	senseWorldFlow[index].addValue(imu_FlowVal[index]);
	senseWorldFlow[index].stlen_( ~dataMeanLengthFlow );
	imu_FlowVal[index]=senseWorldFlow[index].shortMean ;
	if(imu_FlowVal[index]>0.0001) {~synths[index].set(
		\ifeedbk1, ((specFbk1.map(imu_FlowVal[index])*~flowFbkFactor) + (specFbk1.map(imu_EnVal[index])*~enFbkFactor)) * freqFbkFactor[index],
		\ifeedbk2, ((specFbk2.map(imu_FlowVal[index])*~flowFbkFactor) + (specFbk2.map(imu_EnVal[index])*~enFbkFactor)) * freqFbkFactor[index],
		\ibreath, ((specBreath.map(imu_FlowVal[index])*~flowBreathFactor)+ (specBreath.map(imu_EnVal[index])*~enBreathFactor)),
		//\ibreath, ((specBreath.map(imu_FlowVal[index])*~flowBreathFactor)+ (specBreath.map(imu_EnVal[index])*~enBreathFactor)).postln,
		  \vibDepth, specVibDepth.map(imu_FlowVal[index]).midiratio,
		    \vibRate, specVibRate.map(imu_FlowVal[index]),
	    )}  {"nan".postln};
})
};


[  '/imu/1', '/imu/2','/imu/3','/imu/4'].do{ |imu, index|  ~function.add(imu.asSymbol->   {| valor | var specAmp= ControlSpec(~minAmp,~maxAmp,~ampCurve),
		 specLpf= ControlSpec(~minLpf,~maxLpf,0), accelGlobImu=0,imuAngleY=0, imuAngleX=0, forFreqs, forAmps, forQs,globalStillness=1,which,synthGroup1,synthGroup2,pan;

	//valor=valor[1..3].abs.mean.linlin(0,500,0,1);
	accelGlobImu=[35,30,30,35,20,20].collect{|limit, axis| valor[axis +4].abs.linlin(0,limit,0,1)}.maxItem	;
	senseTheWorld[index].addValue(accelGlobImu);
	senseTheWorld[index].stlen_( ~dataMeanLengthEnergy );
	accelGlobImu=senseTheWorld[index].shortMean ;
 	imu_EnVal[index] = accelGlobImu;
   		globalStillness=if(accelGlobImu<~threshold) {~stopValue}{1};
		//valor.postln;
	    ~viewData=accelGlobImu;
		imuAngleY= valor[2];
	freq[index]= imuAngleY.linlin(-90,90,~minPitch[index],~maxPitch[index]);

	freqFbkFactor[index]=freq.linlin(~minPitch[index].midicps,~maxPitch[index].midicps, ~freqFbkFactorMax,~freqFbkFactorMin);
	    imuAngleX=valor[1];
	    pan=valor[1].linlin(0,360,0,~maxPan);


       	~synths[index].set(\freq, freq[index]);
		~synths[index].set(\amp2, specAmp.map(imu_EnVal[index]) *globalStillness );
       ~synths[index].set(\lpf, specLpf.map(imu_EnVal[index]) );
	   //  ~synths[index].set(\panPos,pan);


	}

);
};
};
