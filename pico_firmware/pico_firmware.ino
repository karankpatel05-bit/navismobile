#include <Servo.h>
#include <WebServer.h>
#include <WiFi.h>

const char *ssid = "ACT_2.4G";
const char *password = "18001723";

WebServer server(80);
Servo mouthServo;

// Configuration Pins
const int SERVO_PIN = 15;

// L298N motor driver pins
// Motor A (Left)
const int ENA = 2;
const int IN1 = 3;
const int IN2 = 4;
// Motor B (Right)
const int ENB = 10;
const int IN3 = 11;
const int IN4 = 12;

// Mouth state
bool mouthActive = false;
unsigned long lastFlapTime = 0;
const int FLAP_INTERVAL = 150;
bool mouthOpen = false;

// Periodic IP printer
unsigned long lastIpPrintTime = 0;
const int IP_PRINT_INTERVAL = 5000;

// ─── Motor Functions
// ──────────────────────────────────────────────────────────

void setupMotors() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stopMotors();
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void setMotorSpeeds(int leftSpeed, int rightSpeed) {
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  // Left motor direction
  if (leftSpeed > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else if (leftSpeed < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }

  // Right motor direction
  if (rightSpeed > 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else if (rightSpeed < 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }

  analogWrite(ENA, abs(leftSpeed));
  analogWrite(ENB, abs(rightSpeed));
}

// ─── HTTP Handlers
// ────────────────────────────────────────────────────────────

void handleMove() {
  Serial.print("[MOVE] Request from: ");
  Serial.println(server.client().remoteIP());

  if (!server.hasArg("left") || !server.hasArg("right")) {
    server.send(400, "text/plain", "Missing left/right speed parameters");
    stopMotors();
    return;
  }

  int leftS = server.arg("left").toInt();
  int rightS = server.arg("right").toInt();

  setMotorSpeeds(leftS, rightS);

  server.send(200, "text/plain",
              "OK Left:" + String(leftS) + " Right:" + String(rightS));
}

void handleStop() {
  Serial.println("[STOP] Motors halted via /stop");
  stopMotors();
  server.send(200, "text/plain", "Motors stopped");
}

void handleMouth() {
  Serial.print("[MOUTH] Request from: ");
  Serial.println(server.client().remoteIP());

  if (!server.hasArg("state")) {
    server.send(400, "text/plain", "Missing state parameter");
    return;
  }

  String state = server.arg("state");
  if (state == "start") {
    mouthActive = true;
  } else if (state == "stop") {
    mouthActive = false;
    mouthServo.write(0);
    mouthOpen = false;
  } else {
    server.send(400, "text/plain", "Invalid state. Use 'start' or 'stop'");
    return;
  }

  server.send(200, "text/plain", "Mouth state: " + state);
}

void handleStatus() {
  String json = "{";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"mouthActive\":" + String(mouthActive ? "true" : "false") + ",";
  json += "\"wifiRSSI\":" + String(WiFi.RSSI());
  json += "}";
  server.send(200, "application/json", json);
}

void handleNotFound() { server.send(404, "text/plain", "Route not found"); }

// ─── Setup
// ────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  setupMotors();

  mouthServo.attach(SERVO_PIN);
  mouthServo.write(0);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Register routes
  server.on("/move", handleMove);
  server.on("/stop", handleStop);
  server.on("/mouth", handleMouth);
  server.on("/status", handleStatus);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started");
}

// ─── Loop
// ─────────────────────────────────────────────────────────────────────

void loop() {
  server.handleClient();

  unsigned long currentMillis = millis();

  // Mouth lip-sync flap (non-blocking)
  if (mouthActive) {
    if (currentMillis - lastFlapTime >= FLAP_INTERVAL) {
      lastFlapTime = currentMillis;
      mouthOpen = !mouthOpen;
      mouthServo.write(mouthOpen ? 60 : 0);
    }
  }

  // Periodic IP print
  if (currentMillis - lastIpPrintTime >= IP_PRINT_INTERVAL) {
    lastIpPrintTime = currentMillis;
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("[WARN] WiFi disconnected!");
    }
  }
}