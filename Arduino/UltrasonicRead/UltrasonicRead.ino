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
int t1[numSens];
int t2[numSens];
int t3[numSens];
int t4[numSens];

// Setup
void setup() {
  for(int i = 0; i < numSens * 2; i += 2){
    pinMode(trigMin + i, OUTPUT); 
    pinMode(echoMin + i, INPUT);
  }
  Serial.begin(9600); 
}
void loop() {
  int j = 0;
  for(int i = 0; i < numSens * 2; i += 2){
    digitalWrite(trigMin + i, LOW);
    delayMicroseconds(2);
    digitalWrite(trigMin + i, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigMin + i, LOW);
    duration = pulseIn(echoMin + i, HIGH);
    distance = duration * 0.01723;
    int aux0 = 0;
    int aux1 = 0;
    int aux2 = 0;
    int aux3 = 0;
    aux0 = distance;
    aux1 = t1[j];
    aux2 = t2[j];
    aux3 = t3[j];
    
    distance = (distance + t1[j] + t2[j] + t3[j] + t4[j])/5;
    t1[j] = aux0;
    t2[j] = aux1;
    t3[j] = aux2;
    t4[j] = aux3;
    if (distance > 30 || distance < 5){
      distance = 26;
    }
    Serial.print(distance);
    Serial.print(' ');
    j ++;
  }
  Serial.println();
}
