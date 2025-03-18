
~jointSelectDict=( );

~jointSelectDict.add(\pelvis->0);
~jointSelectDict.add(\l5->1);
~jointSelectDict.add(\l3->2);
~jointSelectDict.add(\t12->3);
~jointSelectDict.add(\sternum->4);
~jointSelectDict.add(\neck->5);
~jointSelectDict.add(\head->6);
~jointSelectDict.add(\rightShoulder->7);
~jointSelectDict.add(\rightUpperArm->8);
~jointSelectDict.add(\rightForeArm->9);
~jointSelectDict.add(\rightHand->10);
~jointSelectDict.add(\leftShoulder->11);
~jointSelectDict.add(\leftUpperArm->12);
~jointSelectDict.add(\leftForeArm->13);
~jointSelectDict.add(\leftHand->14) ;
~jointSelectDict.add(\rightUpperLeg->15) ;
~jointSelectDict.add(\rightLowerLeg->16) ;
~jointSelectDict.add(\rightFoot->17) ;
~jointSelectDict.add(\rightToe->18) ;
~jointSelectDict.add(\leftUpperLeg->19) ;
~jointSelectDict.add(\leftLowerLeg->20) ;
~jointSelectDict.add(\leftFoot->21) ;
~jointSelectDict.add(\leftToe->22) ;

//Select all joints for the mapping fucntion to work
~jointsSelection= (0..22);

//Find either the jojnt index or joint name
~jointSelectDict.findKeyForValue(7);
~jointSelectDict.pelvis;
~jointSelectDict.size;
~jointsSelection.size;





