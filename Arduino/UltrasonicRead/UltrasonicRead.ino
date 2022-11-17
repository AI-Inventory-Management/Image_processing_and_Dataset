/*
  ultrasonic read
    Lee datos de ultrasonicos y envia datos por comunicacion serial
*/



// Constantes
const int trigMin = 22;
const int echoMin = 23;
const int numSens = 8;

// Variables
long duration;
int distance;

// Setup
void setup() {
  for(int i = 0; i < numSens * 2; i += 2){
    pinMode(trigMin + i, OUTPUT); 
    pinMode(echoMin + i, INPUT);
  }
  Serial.begin(9600); 
}
void loop() {
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