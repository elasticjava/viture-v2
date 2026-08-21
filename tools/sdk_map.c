/* Kartiert den V2-Nachrichtenraum: ruft nacheinander alle lesenden SDK-Funktionen
 * auf. Das SDK protokolliert je Aufruf eine Zeile "Built command - MsgID: 0x....";
 * durch Marker im selben Stream ist die Zuordnung eindeutig.
 *
 * Nur Getter - der Geraetezustand wird nicht veraendert.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define PID_PRO2 0x1301
typedef void *H;

static void *lib;
#define SYM(t, v, n) t v = (t)dlsym(lib, n)

typedef H (*f_create)(int);
typedef int (*f_init)(H, const char *, const char *);
typedef int (*f_h)(H);
typedef void (*f_hv)(H);
typedef int (*f_h_u8p)(H, uint8_t *);
typedef int (*f_h_fp)(H, float *);
typedef int (*f_h_cp_ip)(H, char *, int *);

static void mark(const char *name) {
    printf("\n>>>>>> %s\n", name);
    fflush(stdout);
    usleep(250000);
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    lib = dlopen("libglasses.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    SYM(f_create, create, "xr_device_provider_create");
    SYM(f_init, initialize, "xr_device_provider_initialize");
    SYM(f_h, start, "xr_device_provider_start");
    SYM(f_h, stop, "xr_device_provider_stop");
    SYM(f_h, shutdown_, "xr_device_provider_shutdown");
    SYM(f_hv, destroy, "xr_device_provider_destroy");

    H h = create(PID_PRO2);
    if (!h) return 2;
    printf("initialize -> %d\n", initialize(h, NULL, "/tmp"));
    printf("start -> %d\n", start(h));
    usleep(300000);

    /* --- einfache int-Getter ------------------------------------------- */
    const char *simple[] = {
        "xr_device_provider_get_brightness_level",
        "xr_device_provider_get_volume_level",
        "xr_device_provider_get_display_mode",
        "xr_device_provider_get_duty_cycle",
        "xr_device_provider_native_get_mode",
        "xr_device_provider_native_get_dof",
        "xr_device_provider_native_get_display_mode",
        "xr_device_provider_native_get_display_distance",
        "xr_device_provider_native_get_display_size",
        "xr_device_provider_native_get_side_mode",
    };
    for (size_t i = 0; i < sizeof simple / sizeof *simple; i++) {
        SYM(f_h, fn, simple[i]);
        if (!fn) { printf("(fehlt: %s)\n", simple[i]); continue; }
        mark(simple[i]);
        printf("   -> %d\n", fn(h));
    }

    /* --- Getter mit Ausgabeparameter ----------------------------------- */
    SYM(f_h_u8p, get_wear, "xr_device_provider_get_wear_status");
    if (get_wear) {
        mark("xr_device_provider_get_wear_status");
        uint8_t w = 0xAA;
        printf("   -> %d, wear=%u\n", get_wear(h, &w), w);
    }

    SYM(f_h_fp, get_film, "xr_device_provider_get_film_mode");
    if (get_film) {
        mark("xr_device_provider_get_film_mode");
        float v = -1.f;
        printf("   -> %d, voltage=%.3f\n", get_film(h, &v), v);
    }

    SYM(f_h_cp_ip, get_ver, "xr_device_provider_get_glasses_version");
    if (get_ver) {
        mark("xr_device_provider_get_glasses_version");
        char buf[256] = {0};
        int len = (int)sizeof buf;
        int rc = get_ver(h, buf, &len);
        printf("   -> %d, len=%d, version='%s'\n", rc, len, buf);
    }

    SYM(f_h_u8p, get_sn, "xr_device_provider_get_sn_hash");
    if (get_sn) {
        mark("xr_device_provider_get_sn_hash");
        uint8_t hash[32] = {0};
        int rc = get_sn(h, hash);
        printf("   -> %d, hash=", rc);
        for (int i = 0; i < 8; i++) printf("%02x", hash[i]);
        printf("...\n");
    }

    usleep(300000);
    stop(h);
    shutdown_(h);
    destroy(h);
    return 0;
}
