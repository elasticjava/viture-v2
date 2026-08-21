/* Befragt das offizielle Viture-SDK (libglasses.so) zur angeschlossenen Brille
 * und liest anschliessend den IMU-Strom.
 *
 * Phase 1 laeuft ohne Geraetezugriff: die *_is_product_* Funktionen nehmen nur
 * eine Product-ID. Phase 2 oeffnet das Geraet ueber libusb (braucht root).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define PID_PRO2 0x1301

#define IMU_MODE_RAW  0
#define IMU_MODE_POSE 1

typedef void *Handle;
typedef void (*PoseCb)(float *data, uint64_t ts);
typedef void (*RawCb)(float *data, uint64_t ts, uint64_t vsync);
typedef void (*StateCb)(int id, int value);

static Handle (*p_create)(int);
static int (*p_initialize)(Handle, const char *, const char *);
static int (*p_start)(Handle);
static int (*p_stop)(Handle);
static int (*p_shutdown)(Handle);
static void (*p_destroy)(Handle);
static int (*p_reg_pose)(Handle, PoseCb);
static int (*p_reg_raw)(Handle, RawCb);
static int (*p_reg_state)(Handle, StateCb);
static int (*p_open_imu)(Handle, uint8_t, uint8_t);
static int (*p_close_imu)(Handle, uint8_t);
static int (*p_valid)(int);
static int (*p_native_dof)(int);
static int (*p_supports_freq)(int, int, int);
static int (*p_market_name)(int, char *, int *);
static int (*p_devtype)(Handle);

static volatile int pose_n = 0, raw_n = 0;

static void on_pose(float *d, uint64_t ts) {
    if (++pose_n <= 5 || pose_n % 60 == 0)
        printf("  POSE #%-4d roll=%8.2f pitch=%8.2f yaw=%8.2f  q=[%6.3f %6.3f %6.3f %6.3f] ts=%llu\n",
               pose_n, d[0], d[1], d[2], d[3], d[4], d[5], d[6], (unsigned long long)ts);
}

static void on_raw(float *d, uint64_t ts, uint64_t vsync) {
    if (++raw_n <= 5 || raw_n % 60 == 0)
        printf("  RAW  #%-4d [%9.4f %9.4f %9.4f] [%9.4f %9.4f %9.4f] ts=%llu vsync=%llu\n",
               raw_n, d[0], d[1], d[2], d[3], d[4], d[5],
               (unsigned long long)ts, (unsigned long long)vsync);
}

static void on_state(int id, int value) {
    const char *n = id == 0 ? "BRIGHTNESS" : id == 1 ? "VOLUME" : id == 2 ? "DISPLAY_MODE"
                  : id == 3 ? "ELECTROCHROMIC_FILM" : id == 4 ? "NATIVE_DOF"
                  : id == 5 ? "WEAR_STATUS" : "?";
    printf("  STATE id=%d (%s) value=%d\n", id, n, value);
}

#define SYM(var, name)                                                    \
    do {                                                                  \
        *(void **)(&var) = dlsym(lib, name);                              \
        if (!var) { fprintf(stderr, "fehlt: %s\n", name); return 2; }     \
    } while (0)

int main(int argc, char **argv) {
    int do_device = (argc > 1 && strcmp(argv[1], "--device") == 0);
    int raw_mode = (argc > 2 && strcmp(argv[2], "--raw") == 0);

    void *lib = dlopen("libglasses.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    SYM(p_create, "xr_device_provider_create");
    SYM(p_initialize, "xr_device_provider_initialize");
    SYM(p_start, "xr_device_provider_start");
    SYM(p_stop, "xr_device_provider_stop");
    SYM(p_shutdown, "xr_device_provider_shutdown");
    SYM(p_destroy, "xr_device_provider_destroy");
    SYM(p_reg_pose, "xr_device_provider_register_imu_pose_callback");
    SYM(p_reg_raw, "xr_device_provider_register_imu_raw_callback");
    SYM(p_reg_state, "xr_device_provider_register_state_callback");
    SYM(p_open_imu, "xr_device_provider_open_imu");
    SYM(p_close_imu, "xr_device_provider_close_imu");
    SYM(p_valid, "xr_device_provider_is_product_id_valid");
    SYM(p_native_dof, "xr_device_provider_is_product_support_native_dof");
    SYM(p_supports_freq, "xr_device_provider_is_product_support_imu_frequency");
    SYM(p_market_name, "xr_device_provider_get_market_name");
    SYM(p_devtype, "xr_device_provider_get_device_type");

    printf("=== Phase 1: Produktabfragen (ohne Geraetezugriff) ===\n");
    char name[128] = {0};
    int len = (int)sizeof(name);
    printf("PID 0x%04X gueltig:        %d\n", PID_PRO2, p_valid(PID_PRO2));
    if (p_market_name(PID_PRO2, name, &len) == 0)
        printf("Marktname:                %s\n", name);
    printf("Native DOF unterstuetzt:  %d\n", p_native_dof(PID_PRO2));

    static const char *fn[] = {"60Hz", "90Hz", "120Hz", "240Hz", "500Hz", "1000Hz"};
    for (int mode = 0; mode <= 1; mode++) {
        printf("IMU-Modus %-4s Raten:     ", mode == IMU_MODE_RAW ? "RAW" : "POSE");
        for (int f = 0; f < 6; f++)
            printf("%s=%d ", fn[f], p_supports_freq(PID_PRO2, mode, f));
        printf("\n");
    }

    if (!do_device) { printf("\n(Phase 2 uebersprungen; mit --device starten)\n"); return 0; }

    printf("\n=== Phase 2: Geraet oeffnen ===\n");
    Handle h = p_create(PID_PRO2);
    if (!h) { fprintf(stderr, "create() lieferte NULL\n"); return 3; }

    int rc = p_initialize(h, NULL, "/tmp");
    printf("initialize -> %d\n", rc);
    if (rc != 0) { p_destroy(h); return 4; }

    printf("device_type -> %d (0=Gen1 1=Gen2 2=Carina)\n", p_devtype(h));
    p_reg_state(h, on_state);
    if (raw_mode) p_reg_raw(h, on_raw); else p_reg_pose(h, on_pose);

    rc = p_start(h);
    printf("start -> %d\n", rc);

    uint8_t mode = raw_mode ? IMU_MODE_RAW : IMU_MODE_POSE;
    rc = p_open_imu(h, mode, 2 /* 120Hz */);
    printf("open_imu(mode=%d, 120Hz) -> %d\n", mode, rc);

    printf("\n--- 6 Sekunden mitlesen, bitte Kopf bewegen ---\n");
    sleep(6);
    printf("--- pose=%d raw=%d Ereignisse ---\n", pose_n, raw_n);

    p_close_imu(h, mode);
    p_stop(h);
    p_shutdown(h);
    p_destroy(h);
    return 0;
}
