# VITURE Protokoll V2 (Gen2) — USB-HID

Ermittelt an **VITURE Pro 2 XR Glasses**, `VID 0x35CA / PID 0x1301`, Firmware `bcdDevice 0x0200`.
Methode: usbmon-Mitschnitt gegen die Log-Ausgabe des offiziellen SDK (`libglasses.so`), das
jeden Frame mit MsgID, PayloadLen und Checksum beschriftet. Anschließend gegengeprüft durch
eine eigene Implementierung über `hidraw` ohne jede Vendor-Bibliothek.

Reverse Engineering zum Zweck der Interoperabilität mit eigener Hardware.

## Transport

| Eigenschaft | Wert |
|---|---|
| USB | 2.01, Full Speed, 1 Konfiguration, **1 Interface** |
| Interface | Class 03 (HID), SubClass 00, Protocol 00 |
| Endpunkte | `0x01` Interrupt OUT, `0x81` Interrupt IN, je 64 B, Intervall 1 ms |
| HID-Report-Deskriptor | Usage Page `0xFF00`, Usage `0x01`, je ein opaker 64-B-Report IN/OUT, **keine Report-IDs** |

Kommandos werden als kurzes Paket (nur die tatsächliche Framelänge) auf EP OUT geschrieben,
Antworten kommen auf 64 B genullt aufgefüllt zurück. Über `hidraw` ist dem Frame ein
Report-ID-Byte `0x00` voranzustellen.

## Rahmenformat

Alle Felder little-endian.

```
Offset  Größe  Feld
0       2      Präambel, konstant 0x0010
2       2      MsgID
4       2      PayloadLen
6       2      Checksum = Summe aller Payload-Bytes (mod 2^16)
8       n      Payload
```

Gesamtlänge = `8 + PayloadLen`. Die Checksum ist eine simple Bytesumme, kein CRC —
verifiziert an Kommando (`0x01+0x02 = 0x0003`) und Datenpaket (Summe 2204 = `0x089C`).

**ACK-Konvention:** Die Antwort auf ein Kommando trägt `MsgID | 0x2000`.

## Bekannte Nachrichten

### `0x0301` — IMU-Steuerung (Host → Brille)

Payload: 2 Bytes `[stream, rate]`

`stream` ist eine **Bitmaske**, nicht das Enum aus dem SDK-Header:

| Wert | Bedeutung |
|---|---|
| `0x00` | Streams aus |
| `0x01` | Pose-Stream (fusioniertes Quaternion) |
| `0x02` | Raw-Stream (Gyro + Beschleunigung) |

Achtung: Das SDK-Header-Enum nennt `VITURE_IMU_MODE_RAW = 0` und `..._POSE = 1`; auf dem
Draht sind es `2` bzw. `1`. Die API-Werte werden also umgesetzt, nicht durchgereicht.

`rate`: `0`=60 Hz, `1`=90 Hz, `2`=120 Hz, `3`=240 Hz, `4`=500 Hz, `5`=1000 Hz.
Die Pro 2 unterstützt laut SDK-Abfrage Raw bis 1000 Hz, Pose nur bis 240 Hz.

Beispiele:
```
10 00 01 03 02 00 03 00 01 02    Pose-Stream, 120 Hz
10 00 01 03 02 00 04 00 02 02    Raw-Stream,  120 Hz
10 00 01 03 02 00 00 00 00 00    beide aus
```

### `0x2301` — ACK auf `0x0301` (Brille → Host)

Payload: 1 Byte Status, `0x00` = Erfolg.

### `0x7308` — Pose-Ereignis (Brille → Host)

Payload: 24 Bytes

```
0   u32   unklar (immer < 2^16, springt sprunghaft — kein Zähler)
4   u32   Zeitstempel
8   f32   qw
12  f32   qx
16  f32   qy
20  f32   qz
```

Quaternion ist normiert (gemessen |q| = 1.0000). **Roll/Pitch/Yaw werden nicht übertragen** —
die Euler-Winkel der SDK-Callbacks werden hostseitig aus dem Quaternion berechnet.

### `0x7309` — Raw-Ereignis (Brille → Host)

Payload: 56 Bytes

```
0   u32   unklar (wie oben)
4   u32   Zeitstempel (inkrementiert je Sample um 1)
8   u16   unklar, ~186/187, leicht schwankend
10  f32   Gyro X   [rad/s]
14  f32   Gyro Y
18  f32   Gyro Z
22  f32   Accel X  [g]
26  f32   Accel Y
30  f32   Accel Z
34  f32   konstant 117.30   — vermutlich Kalibrier-/Skalenwert
38  f32   konstant  60.75
42  f32   konstant 254.85
46  ...   Rest wechselnd, nicht als f32 plausibel
```

Beachte den Versatz: Bei `0x7308` beginnen die Floats bei Offset 8, bei `0x7309` wegen des
zusätzlichen u16 erst bei Offset 10.

Gegenprobe: Beschleunigungsbetrag `|(0.0024, -0.3221, -0.9287)| = 0.983 g`, und
`atan2(0.322, 0.929) = 19.1°` deckt sich mit dem gleichzeitig gemeldeten Pitch.

## Kein Handshake nötig

`xr_device_provider_initialize()` und `start()` senden **nichts** auf dem Interrupt-Endpunkt.
Der komplette Mitschnitt einer Sitzung besteht aus zwei OUT-Kommandos (Stream an, Stream aus),
zwei ACKs und den Datenpaketen. Ein Klient muss lediglich das Gerät öffnen und ein einziges
10-Byte-Kommando senden.

## Abfragen (Getter)

Ein Getter ist ein Frame mit `PayloadLen = 0`, also acht Byte. Die Antwort trägt
`MsgID + 0x2000` und beginnt mit einem **Statusbyte** (`0x00` = ok), danach folgt der Wert.

| Funktion | MsgID | Antwort-MsgID | Nutzlast der Antwort | gemessen an Pro 2 |
|---|---|---|---|---|
| Seriennummer | `0x3002` | `0x5002` | Status + 15 ASCII | Klartext, **kein Hash** |
| Firmware-Version | `0x3003` | `0x5003` | Status + 20 ASCII | `30.0.00.002_20260804` |
| Helligkeit | `0x3122` | `0x5122` | Status + u8 | 3 |
| Duty-Cycle | `0x3125` | `0x5125` | Status + u8 (Prozent) | 98 |
| Anzeigemodus | `0x3141` | `0x5141` | Status + u8 | `0x31` = 1920×1080@60 |
| Lautstärke | `0x3201` | `0x5201` | Status + u8 | 5 |
| Trage-Status | `0x3321` | `0x5321` | Status + u8 | 0 = nicht getragen |

Beispiel Helligkeit:
```
OUT  10 00 22 31 00 00 00 00
IN   10 00 22 51 02 00 03 00 00 03      → Status 0, Wert 3
```

Die MsgIDs folgen einer Ordnung: `0x30xx` Geräteinformation, `0x31xx` Anzeige,
`0x32xx` Audio, `0x33xx` Sensorik, `0x03xx` IMU-Steuerung, `0x73xx` IMU-Ereignisse.
Setter dürften in denselben Gruppen liegen, sind aber nicht vermessen.

### Zwei Beobachtungen, die für eigene Implementierungen sprechen

Die **Seriennummer geht im Klartext** über den Draht; das SDK bildet den SHA-256 erst
hostseitig, obwohl sein Header behauptet, die rohe Nummer werde nie herausgegeben.

`xr_device_provider_get_wear_status()` liefert `0` als Rückgabewert, **befüllt den
Ausgabeparameter aber nicht** — der Sentinel im Testpuffer blieb unverändert. Auf dem Draht
steht der Wert dagegen sauber da. Direkter Zugriff ist hier also genauer als das Vendor-SDK.

## Anzeigemodi

Konstanten aus `viture_protocol_public.h`, bestätigt durch `0x3141`:

| Wert | Modus |
|---|---|
| `0x31` | 1920×1080 @ 60 Hz |
| `0x32` | 3840×1080 @ 60 Hz (SBS) |
| `0x33` | 1920×1080 @ 90 Hz |
| `0x34` | 1920×1080 @ 120 Hz |
| `0x35` | 3840×1080 @ 90 Hz (SBS) |

Die 1200p-Varianten (`0x41`–`0x45`) gelten für Luma-Modelle.

## Nicht unterstützt auf der Pro 2

`native_get_*` und `get_film_mode` liefern `-4` (`NOT_SUPPORTED`) und senden **gar kein
Kommando** — das bestätigt unabhängig: kein natives DOF, keine Elektrochromfolie.

## Noch offen

Die Bedeutung des jeweils ersten u32 in Ereignispaketen (springt sprunghaft, kein Zähler)
und des u16 an Offset 8 im Raw-Paket. Ebenso die Setter-MsgIDs — die Vermessung ist
identisch, verändert aber Gerätezustand und wurde daher zurückgestellt.

## Referenzimplementierungen

`viture-v2` — Rust-Crate ohne Abhängigkeiten. Protokollkern, Transport als Trait,
`hidraw`-Implementierung für Linux. Der Hot Path allokiert nicht. Fünf Tests prüfen
Frame-Bau und Zerlegung gegen die hier dokumentierten, real mitgeschnittenen Bytes.

Gemessen gegen die Hardware: Pose 478 Ereignisse in 4,0 s (119 Hz), Raw 300 in 3,0 s
(100 Hz), `|a|` konstant 0,985 g. Alle Getter aus der Tabelle oben liefern plausible Werte.

`v2_hidraw.py` / `v2_raw.py` — die Python-Erstimplementierungen, mit denen das Protokoll
zuerst bestätigt wurde.
