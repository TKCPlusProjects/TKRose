#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const int max_iterations = 128;

const double stop_threshold = 0.01;

const double grad_step = 0.01;

const double clip_far = 10.0;

const double PI = 3.14159265359;

const double PI2 = 6.28318530718;

const double DEG_TO_RAD = PI / 180.0;

typedef struct { double x, y; } vec2;

typedef struct { double x, y, z; } vec3;

typedef struct { double m[9]; } mat3;

const vec3 light_pos = { 20.0, 50.0, 20.0 };

double min(double a, double b) { return a < b ? a : b; }

double max(double a, double b) { return a > b ? a : b; }

double clamp(double f, double a, double b) { return max(min(f, b), a); }

vec2 make2(double x, double y) { vec2 r = { x, y }; return r; }

vec2 add2(vec2 a, vec2 b) { vec2 r = { a.x + b.x, a.y + b.y }; return r; }

vec2 sub2(vec2 a, vec2 b) { vec2 r = { a.x - b.x, a.y - b.y }; return r; }

double dot2(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }

double length2(vec2 v) { return sqrt(dot2(v, v)); }

vec3 make3(double x, double y, double z) { vec3 r = { x, y, z }; return r; }

vec3 add3(vec3 a, vec3 b) { vec3 r = { a.x + b.x, a.y + b.y, a.z + b.z }; return r; }

vec3 sub3(vec3 a, vec3 b) { vec3 r = { a.x - b.x, a.y - b.y, a.z - b.z }; return r; }

vec3 mul3(vec3 a, vec3 b) { vec3 r = { a.x * b.x, a.y * b.y, a.z * b.z }; return r; }

vec3 scale3(vec3 v, double s) { vec3 r = { v.x * s, v.y * s, v.z * s }; return r; }

double dot3(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

double length3(vec3 v) { return sqrt(dot3(v, v)); }

vec3 normalize3(vec3 v) { return scale3(v, 1.0 / length3(v)); }

vec3 mul(mat3 m, vec3 v) {
    return make3(
        m.m[0] * v.x + m.m[3] * v.y + m.m[6] * v.z,
        m.m[1] * v.x + m.m[4] * v.y + m.m[7] * v.z,
        m.m[2] * v.x + m.m[5] * v.y + m.m[8] * v.z);
}

mat3 rotationXY(double x, double y) {
    vec2 c = { cos(x), cos(y) };
    vec2 s = { sin(x), sin(y) };
    mat3 m = {
        c.y      , 0.0, -s.y,
        s.y * s.x,  c.x,  c.y * s.x,
        s.y * c.x, -s.x,  c.y * c.x
    };
    return m;
}

double opI(double d1, double d2) { return max(d1, d2); }

double opU(double d1, double d2) { return min(d1, d2); }

double opS(double d1, double d2) { return max(-d1, d2); }

double sdPetal(vec3 p, double s) {
    p = add3(mul3(p, make3(0.8, 1.5, 0.8)), make3(0.1, 0.0, 0.0));
    vec2 q = make2(length2(make2(p.x, p.z)), p.y);

    double lower = length2(q) - 1.0;
    lower = opS(length2(q) - 0.97, lower);

    lower = opI(lower, q.y);
    double upper = length2(sub2(q, make2(s, 0.0))) + 1.0 - s;

    upper = opS(upper, length2(sub2(q, make2(s, 0.0))) + 0.97 - s);

    upper = opI(upper, -q.y);
    upper = opI(upper, q.x - 2.0);
    double region = length3(sub3(p, make3(1.0, 0.0, 0.0))) - 1.0;
    return opI(opU(upper, lower), region);
}

double map(vec3 p) {
    double d = 1000.0, s = 2.0;
    mat3 r = rotationXY(0.1, PI2 * 0.618034);
    r.m[0] *= 1.08;  r.m[1] *= 1.08;  r.m[2] *= 1.08;
    r.m[3] *= 0.995; r.m[4] *= 0.995; r.m[5] *= 0.995;
    r.m[6] *= 1.08;  r.m[7] *= 1.08;  r.m[8] *= 1.08;
    for (int i = 0; i < 21; i++) {
        d = opU(d, sdPetal(p, s));
        p = mul(r, p);
        p = add3(p, make3(0.0, -0.02, 0.0));
        s *= 1.05;
    }
    return d;
}

vec3 gradient(vec3 pos) {
    const vec3 dx = { grad_step, 0.0, 0.0 };
    const vec3 dy = { 0.0, grad_step, 0.0 };
    const vec3 dz = { 0.0, 0.0, grad_step };
    return normalize3(make3(
        map(add3(pos, dx)) - map(sub3(pos, dx)),
        map(add3(pos, dy)) - map(sub3(pos, dy)),
        map(add3(pos, dz)) - map(sub3(pos, dz))));
}

double ray_marching(vec3 origin, vec3 dir, double start, double end) {
    double depth = start;
    for (int i = 0; i < max_iterations; i++) {
        double dist = map(add3(origin, scale3(dir, depth)));
        if (dist < stop_threshold)
            return depth;
        depth += dist * 0.3;
        if (depth >= end)
            return end;
    }
    return end;
}

double shading(vec3 v, vec3 n, vec3 eye) {
    vec3 ev = normalize3(sub3(v, eye));
    vec3 vl = normalize3(sub3(light_pos, v));
    double diffuse = dot3(vl, n) * 0.5 + 0.5;
    double rim = pow(1.0 - max(-dot3(n, ev), 0.0), 2.0) * 0.15;
    double ao = clamp(v.y * 0.5 + 0.5, 0.0, 1.0);
    return (diffuse + rim) * ao;
}

vec3 ray_dir(double fov, vec2 pos) {
    vec3 r = { pos.x, pos.y, -tan((90.0 - fov * 0.5) * DEG_TO_RAD) };
    return normalize3(r);
}

double f(vec2 fragCoord) {
    vec3 dir = ray_dir(45.0, fragCoord);
    vec3 eye = { 0.0, 0.0, 4.5 };
    mat3 rot = rotationXY(-1.0, 1.0);
    dir = mul(rot, dir);
    eye = mul(rot, eye);

    double depth = ray_marching(eye, dir, 0.0, clip_far);
    vec3 pos = add3(eye, scale3(dir, depth));
    if (depth >= clip_far)
        return 0.0;
    else
        return shading(pos, gradient(pos), eye);
}

int main(int argc, char* argv[]) {
	getchar();
	
    for (int y = 0; y < 80; y++) {
        for (int x = 0; x < 160; x++)
            putchar("  .,-:;+=*#@"[(int)(f(make2((x / 160.0 - 0.5) * 2.0, (y / 80.0 - 0.5) * -2.0)) * 12.0)]);
        putchar('\n');
    }
    system("color 0c");

	getchar();
	
    return 0;
}
