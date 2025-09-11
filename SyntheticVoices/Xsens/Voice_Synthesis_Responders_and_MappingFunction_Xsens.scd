
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

~mocap_Soni_fof_A_2_xsens_V3={

var jointsSelection= ~jointsSelection, voices=jointsSelection.size, synths,responders, vowels, reverb, joint= (0*3), senseWorldAcc, senseWorldjerk,accValues=(0..2)!voices,accRotValues= (0..2)!voices;


//senseWorldAcc= SensorData.new !77; //así no funciona la iteración si itero de nuevo
senseWorldAcc=voices.collect{SensorData.new }; //así si
senseWorldjerk=voices.collect{SensorData.new }; //así si

~jerkGlobMeanLength= 1; ~accGlobMeanLength= 10;
~accNorMaxValue=100;
~rotAccNorMaxValue=300;
~jerkNorMaxValue=300;

~lowVoiceA= 'tenorA';~lowVoiceE= 'tenorE';~lowVoiceI= 'tenorI'; ~lowVoiceO= 'tenorO'; ~lowVoiceU= 'tenorU';
~highVoiceA= 'sopranoA';~highVoiceE= 'sopranoE';~highVoiceI= 'sopranoI';~highVoiceO= 'sopranoO';~highVoiceU= 'sopranoU';
~stopValue=0;
~threshold=0.0;
~mocapResp_RotPos.value;
~lowVoice= 'tenorA';
~highVoice= 'sopranoI';

~minPitch=12.midicps ! jointsSelection.size;   ~maxPitch=84.midicps ! jointsSelection.size;
~minAmp=0.0 ! jointsSelection.size; ~maxAmp=0.5 ! jointsSelection.size;~ampCurve=0 ! jointsSelection.size;
~minLpf= 12000 ! jointsSelection.size;  ~maxLpf=19000 ! jointsSelection.size;~lpfCurve=0 ! jointsSelection.size;
~minVibDepth=0 ! jointsSelection.size;~maxVibDepth=1 ! jointsSelection.size; ~vibDepthCurve=1 ! jointsSelection.size;

~revTime=2.0; ~damping= 0.6; ~dryLevel= -4;
vowels=FormantTable.keys.asArray.sort[5..9];
synths= voices.collect{|x|Synth(\FOF_substract_Pulse_8chan, [\freq, 60,\vibDepth, 0.3.midiratio, \vibDepth,0.1.midiratio,\vibRate, 8,\amp, 0.3, \lagAmp, 1,
	\lpfFreq, 19000, \formantFreqs, FormantTable.get('bassA').at(0), \formantAmps, FormantTable.get('bassA').at(1), \formantQs, FormantTable.get('bassA').at(2),
	 \lagFormantFreqs,1, \lagFormantAmps, 1, \lagFormantQs, 1,\out, 50] )};

reverb=8.collect{|x|Synth( \itin_Verb3mono, [\out, 0 +x, \in,50 +x ,\taillevel,-12, \revtime,2.75, \roomsize,143, \damping,0.3, \spread,30.0, \drylevel, -2,  \addAction, 1, \dur, 1], addAction: \addToTail)};

~functionMocap_data_1= (
	'/mocap/joint/pos_world':     {| valor | var  posX=valor[(0,3..valor.size-1)], posY=valor[(1,4..valor.size)], posZ=valor[(joint+2, (joint+5)..valor.size)],minPosZ_Value= 0, maxPosZ_Value= 4,minPosY_Value= -5, maxPosY_Value=5;
		posZ =posZ[jointsSelection];
		posY =posY[jointsSelection];
		(voices).do{|x| synths[x].set(
			//\freq, posZ[x].linlin(minPosZ_Value,maxPosZ_Value,~minPitch[x],~maxPitch[x])* (x.linlin(0,voices,1,1.1)),
			 //\panPos, posY[x].linlin(minPosY_Value,1.0,maxPosY_Value,2.0),
			)
		};
		//pan[0].postln
	},


		'/mocap/joint/rot':     {| valor | var  rotX=valor[(joint+0,(joint+3)..valor.size-1)], rotY=valor[(joint+1,(joint+4)..valor.size)], rotZ=valor[(joint+2, (joint+5)..valor.size)],minRotX_Value= -pi, maxRotX_Value= pi,minRotY_Value= -pi/2, maxRotY_Value= pi/2,minRotZ_Value= -pi, maxRotZ_Value= pi,forFreqs, forAmps, forQs,vowelSpaceWhich,vowelSpacePitch;

		rotX= rotX[jointsSelection];
	    rotY= rotY[jointsSelection];
	    rotZ= rotZ[jointsSelection];
/*vowelSpaceWhich=rotX;
vowelSpacePitch=rotY;*/

		 forFreqs=jointsSelection.size.collect{|x|case(
            {(rotX[x]>= -pi).and(rotX[x]< (-pi/1.5))},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceA).at(0),FormantTable.get(~highVoiceA).at(0))},
            {(rotX[x]>= (-pi/1.5)).and(rotX[x]< (-pi/3))},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceE).at(0),FormantTable.get(~highVoiceE).at(0))},
			{(rotX[x]>= (-pi/3)).and(rotX[x]< 0)}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceI).at(0),FormantTable.get(~highVoiceI).at(0))},
			{(rotX[x]>= 0).and(rotX[x]<(pi/1.5))},  {rotY[x].linlin((pi/2),(pi/2),FormantTable.get(~lowVoiceO).at(0),FormantTable.get(~highVoiceO).at(0))},
			{(rotX[x]>= (pi/1.5)).and(rotX[x]<pi)},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceU).at(0),FormantTable.get(~highVoiceU).at(0))},
		)};

		forAmps=jointsSelection.size.collect{|x| case(
			{(rotX[x]>= -pi).and(rotX[x]< (-pi/1.5))}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceA).at(1),FormantTable.get(~highVoiceA).at(1))},
			{(rotX[x]>= (-pi/1.5)).and(rotX[x]< (-pi/3))}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceE).at(1),FormantTable.get(~highVoiceE).at(1))},
			{(rotX[x]>= (-pi/3)).and(rotX[x]< 0)}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceI).at(1),FormantTable.get(~highVoiceI).at(1))},
			{(rotX[x]>= 0).and(rotX[x]<(pi/1.5))}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceO).at(1),FormantTable.get(~highVoiceO).at(1))},
			{(rotX[x]>= (pi/1.5)).and(rotX[x]<pi)},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceU).at(1),FormantTable.get(~highVoiceU).at(1))},
		)};

		forQs=jointsSelection.size.collect{|x| case(
			{(rotX[x]>= (-pi)).and(rotX[x]< (-pi/1.5))}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceA).at(2),FormantTable.get(~highVoiceA).at(2))},
			{(rotX[x]>= (-pi/1.5)).and(rotX[x]< (-pi/3))},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceE).at(2),FormantTable.get(~highVoiceE).at(2))},
			{(rotX[x]>= (-pi/3)).and(rotX[x]< 0)}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceI).at(2),FormantTable.get(~highVoiceI).at(2))},
			{(rotX[x]>= 0).and(rotX[x]<(pi/1.5))},  {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceO).at(2),FormantTable.get(~highVoiceO).at(2))},
			{(rotX[x]>= (pi/1.5)).and(rotX[x]<pi)}, {rotY[x].linlin((-pi/2),(pi/2),FormantTable.get(~lowVoiceU).at(2),FormantTable.get(~highVoiceU).at(2))},
		)};

		(voices).do{|x| synths[x].set(
			\freq, rotY[x].linlin(minRotY_Value,maxRotY_Value,~minPitch,~maxPitch)* (x.linlin(0,voices,1,1.1)),
			//\freq, rotX[x].linlin(minRotX_Value,maxRotX_Value,~minPitch,~maxPitch)* (x.linlin(0,voices,1,1.1)),
		    \panPos, rotZ[x].linlin(-pi,pi,0.0,2),
			\formantFreqs, forFreqs[x],
			\formantAmps, forAmps[x] ,
			\formantQs,    forQs[x] ,
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
		//~viewData=	means[41];
		//means[41].postln;

		(voices).do{|x| synths[x].set(
			\amp, ControlSpec(~minAmp[x],~maxAmp[x], ~ampCurve[x]).map((means[x]))*globalStillness[x] ,
			\lpfFreq, ControlSpec(~minLpf[x],~maxLpf[x], ~lpfCurve[x]).map((means[x])) ,

                                                  )
		                 };

		//control reverb externally
		8.collect{|x|reverb[x].set(\drylevel, ~dryLevel, \revtime, ~revTime, \damping, ~damping)};
		                                               },

			'/mocap/joint/ang_jerk':     {| valor |var  means, jerkNorMaxValue=500;
		//means=(0,3..valor.size-1).collect{|limit,index| valor[limit..(limit +2)].abs.mean};
	//~viewData=	means[11];
	jerkNorMaxValue= ~jerkNorMaxValue	;
      means=valor.collect{|x|x.linlin(0,jerkNorMaxValue,0,1)}; //lo normalizo
	    means= means[jointsSelection ];
		means.size.collect{|x|senseWorldjerk[x].addValue(means[x])};
		means.size.collect{|x|senseWorldjerk[x].stlen_( ~jerkGlobMeanLength )};
		means.size.collect{|x|means[x]=senseWorldjerk[x].shortMean} ;


				(voices).do{|x| synths[x].set(
			\vibDepth, ControlSpec(~minVibDepth[x],~maxVibDepth[x], ~vibDepthCurve[x]).map((means[x])).midiratio
                                                  )
		                 }

		                                                 },

)
~jointSelectDict.size=  jointsSelection.size;

}