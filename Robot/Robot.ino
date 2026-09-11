#include "CytronMotorDriver.h"
#include <ArduinoJson.h>
#include <string.h>

// Motor trái: PWM = D1(GPIO5), DIR = D2(GPIO4)
CytronMD motor1(PWM_DIR, 5, 4);

// Motor phải: PWM = D5(GPIO14), DIR = D6(GPIO12)
CytronMD motor2(PWM_DIR, 14, 12);

// Cảm biến siêu âm HC-SR04
#define TRIG_PIN 0   // D7
#define ECHO_PIN 2   // D8

// Encoder
#define ENCODER_LEFT_PIN   13   // D3 / GPIO0
#define ENCODER_RIGHT_PIN  15   // D4 / GPIO2

#define PULSES_PER_REV 20.0

// Bánh xe 97mm
#define WHEEL_DIAMETER_MM 97.0
#define WHEEL_CIRCUMFERENCE_CM ((WHEEL_DIAMETER_MM * 3.14159) / 10.0)

volatile long leftPulseAbs = 0;
volatile long rightPulseAbs = 0;
volatile long leftPulseSigned = 0;
volatile long rightPulseSigned = 0;

volatile int leftEncoderDir = 0;
volatile int rightEncoderDir = 0;

#define SPEED_FORWARD 160
#define SPEED_BACK    140
#define SPEED_TURN    160

#define STOP_DISTANCE 35
#define MS_PER_DEGREE 25

#define CAMERA_BASE_SPEED 170
#define CAMERA_TURN_GAIN 120
#define CAMERA_TIMEOUT_MS 5000

String serialBuffer = "";

bool isMoving = false;
bool autoMode = false;
bool cameraMode = false;
unsigned long motionEndTime = 0;
unsigned long lastCameraCmdMs = 0;
unsigned long lastCameraDebugMs = 0;

bool encoderMotionMode = false;
long targetEncoderPulses = 0;

#define TEST_MOTOR_TIME 1500
#define ENCODER_TEST_TIME 10000

bool encoderTestMode = false;
unsigned long encoderTestEndTime = 0;
unsigned long lastEncoderTestPrint = 0;

void ICACHE_RAM_ATTR leftEncoderISR() {
  leftPulseAbs++;

  if (leftEncoderDir > 0) {
    leftPulseSigned++;
  } else if (leftEncoderDir < 0) {
    leftPulseSigned--;
  }
}

void ICACHE_RAM_ATTR rightEncoderISR() {
  rightPulseAbs++;

  if (rightEncoderDir > 0) {
    rightPulseSigned++;
  } else if (rightEncoderDir < 0) {
    rightPulseSigned--;
  }
}

void resetEncoderCounter() {
  noInterrupts();
  leftPulseAbs = 0;
  rightPulseAbs = 0;
  leftPulseSigned = 0;
  rightPulseSigned = 0;
  interrupts();
}

void readEncoderCounter(long &lAbs, long &rAbs, long &lSigned, long &rSigned) {
  noInterrupts();
  lAbs = leftPulseAbs;
  rAbs = rightPulseAbs;
  lSigned = leftPulseSigned;
  rSigned = rightPulseSigned;
  interrupts();
}

long getAverageAbsPulse() {
  long lAbs, rAbs, lSigned, rSigned;
  readEncoderCounter(lAbs, rAbs, lSigned, rSigned);
  return (labs(lAbs) + labs(rAbs)) / 2;
}

float pulseToRound(long pulse) {
  return (float)pulse / PULSES_PER_REV;
}

long roundToPulse(float rounds) {
  return (long)(rounds * PULSES_PER_REV);
}

float cmToRounds(float cm) {
  return cm / WHEEL_CIRCUMFERENCE_CM;
}

float roundsToCm(float rounds) {
  return rounds * WHEEL_CIRCUMFERENCE_CM;
}

void printWheelInfo() {
  Serial.println("========== WHEEL INFO ==========");
  Serial.print("WHEEL_DIAMETER_MM: ");
  Serial.println(WHEEL_DIAMETER_MM, 2);
  Serial.print("WHEEL_CIRCUMFERENCE_CM: ");
  Serial.println(WHEEL_CIRCUMFERENCE_CM, 3);
  Serial.print("PULSES_PER_REV: ");
  Serial.println(PULSES_PER_REV, 2);
  Serial.println("1 vong banh xe di duoc khoang 30.47 cm");
  Serial.println("================================");
}

void printEncoderData() {
  long lAbs, rAbs, lSigned, rSigned;
  readEncoderCounter(lAbs, rAbs, lSigned, rSigned);

  float leftRoundsAbs = pulseToRound(lAbs);
  float rightRoundsAbs = pulseToRound(rAbs);
  float leftRoundsSigned = pulseToRound(lSigned);
  float rightRoundsSigned = pulseToRound(rSigned);

  Serial.print("ENCODER_PULSE_ABS: L=");
  Serial.print(lAbs);
  Serial.print(" R=");
  Serial.println(rAbs);

  Serial.print("ENCODER_ROUND_ABS: L=");
  Serial.print(leftRoundsAbs, 3);
  Serial.print(" R=");
  Serial.println(rightRoundsAbs, 3);

  Serial.print("DISTANCE_CM_ABS: L=");
  Serial.print(roundsToCm(leftRoundsAbs), 2);
  Serial.print(" R=");
  Serial.println(roundsToCm(rightRoundsAbs), 2);

  Serial.print("ENCODER_PULSE_SIGNED: L=");
  Serial.print(lSigned);
  Serial.print(" R=");
  Serial.println(rSigned);

  Serial.print("ENCODER_ROUND_SIGNED: L=");
  Serial.print(leftRoundsSigned, 3);
  Serial.print(" R=");
  Serial.println(rightRoundsSigned, 3);

  Serial.print("DISTANCE_CM_SIGNED: L=");
  Serial.print(roundsToCm(leftRoundsSigned), 2);
  Serial.print(" R=");
  Serial.println(roundsToCm(rightRoundsSigned), 2);
}

void stopCar() {
  motor1.setSpeed(0);
  motor2.setSpeed(0);

  leftEncoderDir = 0;
  rightEncoderDir = 0;

  isMoving = false;
  cameraMode = false;
  encoderMotionMode = false;
  encoderTestMode = false;
}

void setCameraMotorSpeeds(int leftSpeed, int rightSpeed) {
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  if (leftSpeed > 0) {
    leftEncoderDir = 1;
  } else if (leftSpeed < 0) {
    leftEncoderDir = -1;
  } else {
    leftEncoderDir = 0;
  }

  if (rightSpeed > 0) {
    rightEncoderDir = 1;
  } else if (rightSpeed < 0) {
    rightEncoderDir = -1;
  } else {
    rightEncoderDir = 0;
  }

  motor1.setSpeed(leftSpeed);
  motor2.setSpeed(rightSpeed);
}

int normalizedCameraMotorToPwm(float value) {
  value = constrain(value, -1.0, 1.0);
  return (int)(value * CAMERA_BASE_SPEED);
}

void driveByCameraMotors(float leftMotor, float rightMotor, const char* motionMode, bool emergencyStop) {
  cameraMode = true;
  lastCameraCmdMs = millis();

  autoMode = false;
  isMoving = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  if (emergencyStop) {
    if (millis() - lastCameraDebugMs >= 500) {
      lastCameraDebugMs = millis();
      Serial.println("CAMERA EMERGENCY_STOP");
    }
    stopCar();
    return;
  }

  int leftSpeed;
  int rightSpeed;

  if (strcmp(motionMode, "AVOIDANCE_TURN") == 0) {
    leftSpeed = leftMotor < 0 ? -SPEED_TURN : SPEED_TURN;
    rightSpeed = rightMotor < 0 ? -SPEED_TURN : SPEED_TURN;
  } else {
    leftSpeed = normalizedCameraMotorToPwm(leftMotor);
    rightSpeed = normalizedCameraMotorToPwm(rightMotor);
  }

  setCameraMotorSpeeds(leftSpeed, rightSpeed);

  if (millis() - lastCameraDebugMs >= 500) {
    lastCameraDebugMs = millis();
    Serial.print("CAMERA motor L=");
    Serial.print(leftMotor, 3);
    Serial.print(" R=");
    Serial.print(rightMotor, 3);
    Serial.print(" pwmL=");
    Serial.print(leftSpeed);
    Serial.print(" pwmR=");
    Serial.print(rightSpeed);
    Serial.print(" mode=");
    Serial.println(motionMode);
  }
}

void driveByCameraJson(float steer, bool emergencyStop) {
  cameraMode = true;
  lastCameraCmdMs = millis();

  autoMode = false;
  isMoving = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  if (emergencyStop) {
    if (millis() - lastCameraDebugMs >= 500) {
      lastCameraDebugMs = millis();
      Serial.println("CAMERA EMERGENCY_STOP");
    }
    stopCar();
    return;
  }

  if (steer > 1.0 || steer < -1.0) {
    int turnSpeed = steer > 0 ? SPEED_TURN : -SPEED_TURN;
    setCameraMotorSpeeds(turnSpeed, -turnSpeed);

    if (millis() - lastCameraDebugMs >= 500) {
      lastCameraDebugMs = millis();
      Serial.print("CAMERA tank steer=");
      Serial.print(steer, 3);
      Serial.print(" left=");
      Serial.print(turnSpeed);
      Serial.print(" right=");
      Serial.println(-turnSpeed);
    }
    return;
  }

  steer = constrain(steer, -1.0, 1.0);

  int turn = (int)(steer * CAMERA_TURN_GAIN);
  int leftSpeed = constrain(CAMERA_BASE_SPEED + turn, 0, 255);
  int rightSpeed = constrain(CAMERA_BASE_SPEED - turn, 0, 255);

  setCameraMotorSpeeds(leftSpeed, rightSpeed);

  if (millis() - lastCameraDebugMs >= 500) {
    lastCameraDebugMs = millis();
    Serial.print("CAMERA steer=");
    Serial.print(steer, 3);
    Serial.print(" left=");
    Serial.print(leftSpeed);
    Serial.print(" right=");
    Serial.println(rightSpeed);
  }
}

void startCameraMotorTest(float steer) {
  cameraMode = false;
  autoMode = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  steer = constrain(steer, -1.0, 1.0);

  int turn = (int)(steer * CAMERA_TURN_GAIN);
  int leftSpeed = constrain(CAMERA_BASE_SPEED + turn, 0, 255);
  int rightSpeed = constrain(CAMERA_BASE_SPEED - turn, 0, 255);

  leftEncoderDir = 1;
  rightEncoderDir = 1;

  motor1.setSpeed(leftSpeed);
  motor2.setSpeed(rightSpeed);

  isMoving = true;
  motionEndTime = millis() + TEST_MOTOR_TIME;

  Serial.print("CAMERA_TEST steer=");
  Serial.print(steer, 3);
  Serial.print(" left=");
  Serial.print(leftSpeed);
  Serial.print(" right=");
  Serial.println(rightSpeed);
}

void updateCameraTimeout() {
  if (cameraMode && millis() - lastCameraCmdMs > CAMERA_TIMEOUT_MS) {
    stopCar();
    cameraMode = false;
    Serial.println("CAMERA TIMEOUT STOP");
  }
}

void forwardMotor() {
  leftEncoderDir = 1;
  rightEncoderDir = 1;

  motor1.setSpeed(SPEED_FORWARD);
  motor2.setSpeed(SPEED_FORWARD);
}

void backMotor() {
  leftEncoderDir = -1;
  rightEncoderDir = -1;

  motor1.setSpeed(-SPEED_BACK);
  motor2.setSpeed(-SPEED_BACK);
}

void turnRightMotor() {
  leftEncoderDir = 1;
  rightEncoderDir = -1;

  motor1.setSpeed(SPEED_TURN);
  motor2.setSpeed(-SPEED_TURN);
}

void turnLeftMotor() {
  leftEncoderDir = -1;
  rightEncoderDir = 1;

  motor1.setSpeed(-SPEED_TURN);
  motor2.setSpeed(SPEED_TURN);
}

long readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return 999;
  }

  return duration * 0.034 / 2;
}

void startForwardCM(float cm) {
  if (cm <= 0) {
    Serial.println("ERROR: cm khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = true;
  encoderTestMode = false;

  resetEncoderCounter();

  float rounds = cmToRounds(cm);
  targetEncoderPulses = roundToPulse(rounds);

  Serial.print("ACK: forward ");
  Serial.print(cm, 2);
  Serial.print(" cm = ");
  Serial.print(rounds, 3);
  Serial.print(" rounds, target pulse = ");
  Serial.println(targetEncoderPulses);

  forwardMotor();

  isMoving = true;
  motionEndTime = millis() + (unsigned long)(rounds * 3000) + 5000;
}

void startBackCM(float cm) {
  if (cm <= 0) {
    Serial.println("ERROR: cm khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = true;
  encoderTestMode = false;

  resetEncoderCounter();

  float rounds = cmToRounds(cm);
  targetEncoderPulses = roundToPulse(rounds);

  Serial.print("ACK: back ");
  Serial.print(cm, 2);
  Serial.print(" cm = ");
  Serial.print(rounds, 3);
  Serial.print(" rounds, target pulse = ");
  Serial.println(targetEncoderPulses);

  backMotor();

  isMoving = true;
  motionEndTime = millis() + (unsigned long)(rounds * 3000) + 5000;
}

void startRightAngle(int angle) {
  if (angle <= 0) {
    Serial.println("ERROR: goc khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  unsigned long turnTime = (unsigned long)angle * MS_PER_DEGREE;

  Serial.print("ACK: right ");
  Serial.print(angle);
  Serial.println(" degree");

  resetEncoderCounter();
  turnRightMotor();

  isMoving = true;
  motionEndTime = millis() + turnTime;
}

void startLeftAngle(int angle) {
  if (angle <= 0) {
    Serial.println("ERROR: goc khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  unsigned long turnTime = (unsigned long)angle * MS_PER_DEGREE;

  Serial.print("ACK: left ");
  Serial.print(angle);
  Serial.println(" degree");

  resetEncoderCounter();
  turnLeftMotor();

  isMoving = true;
  motionEndTime = millis() + turnTime;
}

void startForwardRounds(float rounds) {
  if (rounds <= 0) {
    Serial.println("ERROR: so vong khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = true;
  encoderTestMode = false;

  resetEncoderCounter();

  targetEncoderPulses = roundToPulse(rounds);

  Serial.print("ACK: forward_round ");
  Serial.print(rounds, 3);
  Serial.print(" rounds = ");
  Serial.print(roundsToCm(rounds), 2);
  Serial.print(" cm, target pulse = ");
  Serial.println(targetEncoderPulses);

  forwardMotor();

  isMoving = true;
  motionEndTime = millis() + (unsigned long)(rounds * 3000) + 5000;
}

void startBackRounds(float rounds) {
  if (rounds <= 0) {
    Serial.println("ERROR: so vong khong hop le");
    Serial.println("DONE");
    return;
  }

  autoMode = false;
  encoderMotionMode = true;
  encoderTestMode = false;

  resetEncoderCounter();

  targetEncoderPulses = roundToPulse(rounds);

  Serial.print("ACK: back_round ");
  Serial.print(rounds, 3);
  Serial.print(" rounds = ");
  Serial.print(roundsToCm(rounds), 2);
  Serial.print(" cm, target pulse = ");
  Serial.println(targetEncoderPulses);

  backMotor();

  isMoving = true;
  motionEndTime = millis() + (unsigned long)(rounds * 3000) + 5000;
}

void updateMotion() {
  if (!isMoving) return;

  if (encoderMotionMode) {
    long avgPulse = getAverageAbsPulse();

    if (avgPulse >= targetEncoderPulses) {
      stopCar();
      printEncoderData();
      Serial.println("DONE");
      return;
    }

    if (millis() >= motionEndTime) {
      stopCar();
      Serial.println("ERROR: encoder timeout, kiem tra day encoder hoac PULSES_PER_REV");
      printEncoderData();
      Serial.println("DONE");
      return;
    }
  } else {
    if (millis() >= motionEndTime) {
      stopCar();
      printEncoderData();
      Serial.println("DONE");
    }
  }
}

void startMotorTest(const char* name, int m1Speed, int m2Speed, int leftDir, int rightDir) {
  autoMode = false;
  encoderMotionMode = false;
  encoderTestMode = false;

  resetEncoderCounter();

  leftEncoderDir = leftDir;
  rightEncoderDir = rightDir;

  motor1.setSpeed(m1Speed);
  motor2.setSpeed(m2Speed);

  isMoving = true;
  motionEndTime = millis() + TEST_MOTOR_TIME;

  Serial.print("TEST MOTOR: ");
  Serial.println(name);
}

void startEncoderTest() {
  stopCar();
  resetEncoderCounter();

  encoderTestMode = true;
  encoderTestEndTime = millis() + ENCODER_TEST_TIME;
  lastEncoderTestPrint = 0;

  Serial.println("TEST ENCODER BAT DAU");
  Serial.println("Hay quay tung banh xe bang tay trong 10 giay.");
}

void updateEncoderTest() {
  if (!encoderTestMode) return;

  if (millis() - lastEncoderTestPrint >= 500) {
    lastEncoderTestPrint = millis();
    printEncoderData();
    Serial.println("---");
  }

  if (millis() >= encoderTestEndTime) {
    encoderTestMode = false;
    Serial.println("TEST ENCODER KET THUC");
    printEncoderData();
    Serial.println("DONE");
  }
}

void autoRun() {
  static unsigned long lastCheck = 0;

  if (millis() - lastCheck < 150) return;

  lastCheck = millis();

  long distance = readDistanceCM();

  Serial.print("DISTANCE: ");
  Serial.print(distance);
  Serial.println(" cm");

  if (distance > STOP_DISTANCE) {
    forwardMotor();
  } else {
    stopCar();
    autoMode = false;

    Serial.println("OBSTACLE: STOP");
    printEncoderData();
    Serial.println("DONE");
  }
}

void printHelp() {
  Serial.println();
  Serial.println("========== SERIAL TEST MENU ==========");
  Serial.println("wheel             : xem thong tin banh xe");
  Serial.println("test_encoder / te : test encoder");
  Serial.println("test_forward / tf : tien 1.5 giay");
  Serial.println("test_back / tb    : lui 1.5 giay");
  Serial.println("test_left / tl    : xoay trai 1.5 giay");
  Serial.println("test_right / tr   : xoay phai 1.5 giay");
  Serial.println("f 30              : tien 30 cm bang encoder");
  Serial.println("b 20              : lui 20 cm bang encoder");
  Serial.println("fr 2              : tien 2 vong banh xe");
  Serial.println("br 1.5            : lui 1.5 vong banh xe");
  Serial.println("l 90              : xoay trai 90 do");
  Serial.println("r 90              : xoay phai 90 do");
  Serial.println("encoder / e       : xem encoder");
  Serial.println("reset             : reset encoder");
  Serial.println("distance / d      : doc sieu am");
  Serial.println("stop / s / off    : dung xe");
  Serial.println("on                : bat tu hanh");
  Serial.println("camera_test 0     : test motor bang toc do camera");
  Serial.println("{\"cmd\":\"json_ping\"} : test ESP nhan JSON");
  Serial.println("camera JSON       : {\"steering_value\":0.0,\"left_motor\":1.0,\"right_motor\":1.0,\"turn_type\":\"NORMAL_ADJUSTMENT\"}");
  Serial.println("======================================");
}

void processTextCommand(String line) {
  line.trim();
  cameraMode = false;

  int spaceIndex = line.indexOf(' ');
  String cmd;
  String arg;

  if (spaceIndex >= 0) {
    cmd = line.substring(0, spaceIndex);
    arg = line.substring(spaceIndex + 1);
  } else {
    cmd = line;
    arg = "";
  }

  cmd.trim();
  arg.trim();
  cmd.toLowerCase();

  int valueInt = arg.toInt();
  float valueFloat = arg.toFloat();

  if (cmd == "help" || cmd == "menu" || cmd == "?") {
    printHelp();
  } else if (cmd == "wheel") {
    printWheelInfo();
    Serial.println("DONE");
  } else if (cmd == "test_encoder" || cmd == "te") {
    startEncoderTest();
  } else if (cmd == "test_forward" || cmd == "tf") {
    startMotorTest("FORWARD", SPEED_FORWARD, SPEED_FORWARD, 1, 1);
  } else if (cmd == "test_back" || cmd == "tb") {
    startMotorTest("BACK", -SPEED_BACK, -SPEED_BACK, -1, -1);
  } else if (cmd == "test_left" || cmd == "tl") {
    startMotorTest("TURN LEFT", -SPEED_TURN, SPEED_TURN, -1, 1);
  } else if (cmd == "test_right" || cmd == "tr") {
    startMotorTest("TURN RIGHT", SPEED_TURN, -SPEED_TURN, 1, -1);
  } else if (cmd == "camera_test" || cmd == "ct") {
    startCameraMotorTest(valueFloat);
  } else if (cmd == "f" || cmd == "forward") {
    startForwardCM(valueFloat);
  } else if (cmd == "b" || cmd == "back") {
    startBackCM(valueFloat);
  } else if (cmd == "l" || cmd == "left") {
    startLeftAngle(valueInt);
  } else if (cmd == "r" || cmd == "right") {
    startRightAngle(valueInt);
  } else if (cmd == "fr" || cmd == "forward_round") {
    startForwardRounds(valueFloat);
  } else if (cmd == "br" || cmd == "back_round") {
    startBackRounds(valueFloat);
  } else if (cmd == "encoder" || cmd == "e") {
    printEncoderData();
    Serial.println("DONE");
  } else if (cmd == "reset") {
    resetEncoderCounter();
    Serial.println("ENCODER RESET");
    Serial.println("DONE");
  } else if (cmd == "distance" || cmd == "d") {
    long dist = readDistanceCM();
    Serial.print("DISTANCE: ");
    Serial.print(dist);
    Serial.println(" cm");
    Serial.println("DONE");
  } else if (cmd == "stop" || cmd == "s" || cmd == "off") {
    autoMode = false;
    stopCar();
    Serial.println("STOP");
    printEncoderData();
    Serial.println("DONE");
  } else if (cmd == "on") {
    autoMode = true;
    isMoving = false;
    encoderMotionMode = false;
    encoderTestMode = false;
    Serial.println("AUTO MODE ON");
  } else {
    Serial.println("ERROR: lenh khong hop le. Go help de xem lenh.");
    Serial.println("DONE");
  }
}

void processJsonCommand(String jsonLine) {
  StaticJsonDocument<512> doc;

  DeserializationError error = deserializeJson(doc, jsonLine);

  if (error) {
    Serial.println("ERROR: JSON sai dinh dang");
    Serial.println("DONE");
    return;
  }

  if (doc.containsKey("steering_value")) {
    long frameIdx = doc["frame_idx"] | -1;
    float steer = doc["steering_value"] | 0.0;
    float leftMotor = doc["left_motor"] | 0.0;
    float rightMotor = doc["right_motor"] | 0.0;
    const char* turnType = doc["turn_type"] | "";
    const char* motionMode = doc["motion_mode"] | "";
    bool emergencyStop = strcmp(turnType, "EMERGENCY_STOP") == 0;
    bool hasDirectMotors = doc.containsKey("left_motor") && doc.containsKey("right_motor");

    Serial.print("JSON_CAMERA_OK frame=");
    Serial.print(frameIdx);
    Serial.print(" steer=");
    Serial.print(steer, 3);
    Serial.print(" left_motor=");
    Serial.print(leftMotor, 3);
    Serial.print(" right_motor=");
    Serial.print(rightMotor, 3);
    Serial.print(" mode=");
    Serial.print(motionMode);
    Serial.print(" emergency=");
    Serial.println(emergencyStop ? 1 : 0);

    if (hasDirectMotors) {
      driveByCameraMotors(leftMotor, rightMotor, motionMode, emergencyStop);
    } else {
      driveByCameraJson(steer, emergencyStop);
    }
    return;
  }

  cameraMode = false;

  const char* cmdChar = doc["cmd"] | "";
  float valueFloat = doc["value"] | 0.0;
  int valueInt = (int)valueFloat;

  String cmd = String(cmdChar);
  cmd.toLowerCase();

  if (cmd == "forward" || cmd == "f") {
    startForwardCM(valueFloat);
  } else if (cmd == "json_ping") {
    Serial.println("JSON_PONG");
    Serial.println("DONE");
  } else if (cmd == "camera_test" || cmd == "ct") {
    startCameraMotorTest(valueFloat);
  } else if (cmd == "back" || cmd == "b") {
    startBackCM(valueFloat);
  } else if (cmd == "right" || cmd == "r") {
    startRightAngle(valueInt);
  } else if (cmd == "left" || cmd == "l") {
    startLeftAngle(valueInt);
  } else if (cmd == "forward_round" || cmd == "fr") {
    startForwardRounds(valueFloat);
  } else if (cmd == "back_round" || cmd == "br") {
    startBackRounds(valueFloat);
  } else if (cmd == "encoder") {
    printEncoderData();
    Serial.println("DONE");
  } else if (cmd == "wheel") {
    printWheelInfo();
    Serial.println("DONE");
  } else if (cmd == "reset_encoder") {
    resetEncoderCounter();
    Serial.println("ENCODER RESET");
    Serial.println("DONE");
  } else if (cmd == "distance") {
    long d = readDistanceCM();
    Serial.print("DISTANCE: ");
    Serial.print(d);
    Serial.println(" cm");
    Serial.println("DONE");
  } else if (cmd == "stop" || cmd == "off") {
    autoMode = false;
    stopCar();
    Serial.println("STOP");
    Serial.println("DONE");
  } else if (cmd == "on") {
    autoMode = true;
    Serial.println("AUTO MODE ON");
  } else {
    Serial.println("ERROR: lenh JSON khong hop le");
    Serial.println("DONE");
  }
}

void processSerialCommand(String line) {
  line.trim();

  if (line.length() == 0) return;

  if (line.startsWith("{")) {
    processJsonCommand(line);
  } else {
    processTextCommand(line);
  }
}

void readSerialContinuous() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        processSerialCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(ENCODER_LEFT_PIN, INPUT_PULLUP);
  pinMode(ENCODER_RIGHT_PIN, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_LEFT_PIN), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCODER_RIGHT_PIN), rightEncoderISR, RISING);

  resetEncoderCounter();
  stopCar();

  Serial.println("ESP8266 READY");
  Serial.println("Banh xe 97mm da duoc cai dat");
  Serial.println("Go help de xem danh sach lenh");

  printWheelInfo();
  printHelp();
}

void loop() {
  readSerialContinuous();
  updateCameraTimeout();

  if (autoMode) {
    autoRun();
  }

  updateEncoderTest();
  updateMotion();
}
