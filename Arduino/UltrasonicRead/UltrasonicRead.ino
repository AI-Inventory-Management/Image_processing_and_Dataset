/*
Read Ultrasonic Sensors.

Functions:
    setup()
    
    loop()

Author
    Alejandro Dominguez
*/



// Constants
const int trigMin = 22;
const int echoMin = 23;
const int numSens = 8;

// Variables
long duration;
int distance;

// Setup
void setup() {
  // Initalize
  for(int i = 0; i < numSens * 2; i += 2){
    pinMode(trigMin + i, OUTPUT); 
    pinMode(echoMin + i, INPUT);
  }
  
  Serial.begin(9600); 
}

// Main
void loop() {
  /*
    Read each ultrasonic sensor, calculate the distance in centimiters and send
    the information through serial communication.
  */
  for(int i = 0; i < numSens * 2; i += 2){
    digitalWrite(trigMin + i, LOW);
    delayMicroseconds(2);
    digitalWrite(trigMin + i, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigMin + i, LOW);
    duration = pulseIn(echoMin + i, HIGH);
    distance = duration * 0.034 / 2;
    Serial.print(distance);
    Serial.print(' ');
  }
  
  Serial.println();
}