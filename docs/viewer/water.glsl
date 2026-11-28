// Animated water, shared by scripts/render-scene.py --animate and docs/viewer/viewer.js.
// The host defines   vec3 water_tex(vec2 uv, float bias)   (the photographed water, linear
// RGB; bias is a mip bias, so a large one is the photograph's low frequencies) and calls
// water_colour(). Everything is in the renderer's world frame (metres; y up).
//
// The photograph's broad shapes stay where they are -- its reflections, its foam and its
// colour belong to places in the world -- while its fine ripples are carried along the
// current (the high frequencies only, two copies half a cycle apart, each spot on its own
// clock, the contrast held constant), and a moving surface is laid over it:
//  * ripples carried by the current (a river's boils, two copies advected along the flow
//    field half a cycle apart, each spot on its own clock so nothing resets at once, blended
//    so the contrast stays constant),
//  * wind waves (directional, deep-water dispersion, Doppler-shifted by the current),
//  * the swell toward the shore at sea,
// which tilt the surface: the photograph is shifted a little by the slope (it refracts and
// reflects), brightened or darkened as the tilt turns it toward a darker or paler part of
// the sky, and glints where a facet catches the sun and the photograph is already bright.
// At sea the photographed foam is re-timed: it shows where a breaking front passes, and
// broken foam and the swash at the waterline run in with the fronts and back.
// Boats underway carry their wake (a turbulent trail and the Kelvin arms) and a bow wave.

float wa_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float wa_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(wa_hash(i), wa_hash(i + vec2(1.0, 0.0)), u.x),
               mix(wa_hash(i + vec2(0.0, 1.0)), wa_hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Value noise and its gradient.
vec3 wa_noised(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = wa_hash(i), b = wa_hash(i + vec2(1.0, 0.0)), c = wa_hash(i + vec2(0.0, 1.0)), d = wa_hash(i + vec2(1.0, 1.0));
    float k = a - b - c + d;
    return vec3(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y,
                du * vec2(b - a + k * u.y, c - a + k * u.x));
}

// The slope of the boils and streaks a current carries, at q in the current's own frame
// (x along it, stretched: a river's surface is drawn out downstream).
vec2 wa_boils(vec2 q, vec2 f, vec2 sd) {
    vec2 a = vec2(dot(q, f) * 0.5, dot(q, sd));
    vec3 n1 = wa_noised(a * 0.45);
    vec3 n2 = wa_noised(a * 1.4 + 7.0);
    vec2 g = n1.yz * 0.45 * 0.16 + n2.yz * 1.4 * 0.05;
    return f * g.x * 0.5 + sd * g.y;
}

// The slope of the current-borne ripples at p: two copies carried along the current half a
// cycle apart, each spot on its own clock, blended so the contrast stays the same.
vec2 wa_current(vec2 p, vec2 vel, float t) {
    float T = 8.0;
    float sp = length(vel);
    vec2 f = sp > 1e-4 ? vel / sp : vec2(0.0, 1.0);
    vec2 sd = vec2(-f.y, f.x);
    float j = wa_noise(p * 0.06);
    float f0 = fract(t / T + j), f1 = fract(t / T + j + 0.5);
    vec2 s0 = wa_boils(p - vel * (f0 - 0.5) * T, f, sd);
    vec2 s1 = wa_boils(p - vel * (f1 - 0.5) * T + vec2(37.1, 11.3), f, sd);
    float w = 1.0 - abs(2.0 * f0 - 1.0);
    return (s0 * w + s1 * (1.0 - w)) / sqrt(w * w + (1.0 - w) * (1.0 - w));
}

// The slope of the wind's waves at p: six trains about the wind, 0.35 to 3 m long.
vec2 wa_wind(vec2 p, vec2 vel, vec2 wdir, float wspeed, float t) {
    vec2 s = vec2(0.0);
    float steep = 0.035 + 0.06 * clamp(wspeed / 6.0, 0.0, 1.0);
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        float ang = (wa_hash(vec2(fi, 3.7)) - 0.5) * 1.6;
        vec2 d = vec2(cos(ang) * wdir.x - sin(ang) * wdir.y, sin(ang) * wdir.x + cos(ang) * wdir.y);
        float lam = 0.35 * pow(1.55, fi);
        float k = 6.2831853 / lam;
        float om = sqrt(9.81 * k);
        float ph = k * dot(d, p) - (om + k * dot(d, vel)) * t + 6.2831853 * wa_hash(vec2(fi, 9.1));
        s += d * steep * cos(ph) * (0.6 + 0.4 * wa_noise(p * 0.15 + fi));
    }
    return s;
}

// Wakes. A[i] = (x, z, heading x, heading z), B[i] = (length, beam, speed, on).
void wa_wakes(vec2 p, float t, mat4 A, mat4 B, inout vec2 slope, inout float foam) {
    for (int i = 0; i < 4; i++) {
        if (B[i].w < 0.5) continue;
        vec2 c = A[i].xy;
        vec2 f = normalize(A[i].zw + vec2(1e-6, 0.0));
        vec2 sd = vec2(-f.y, f.x);
        float L = max(B[i].x, 2.0), bm = max(B[i].y, 1.0);
        float k = clamp(B[i].z / 4.0, 0.25, 1.2);
        vec2 q = p - (c - f * 0.5 * L);
        float s = -dot(q, f);                       // metres behind the stern
        float side = dot(q, sd);
        float d = abs(side);
        float along = dot(p - c, f);
        float hull = smoothstep(1.8, 0.0, abs(d - 0.5 * bm - 0.4)) * smoothstep(-0.5 * L - 2.0, 0.3 * L, along)
                   * (1.0 - smoothstep(0.5 * L, 0.5 * L + 2.0, along));
        float bow = smoothstep(0.35 * bm + 2.5, 0.0, length(p - (c + f * 0.5 * L)));
        float w = max(hull * 0.75, bow);
        if (s > 0.0) {
            float spread = 0.5 * bm + 0.1 * s;
            float trail = exp(-s / (1.4 * L)) * smoothstep(spread, 0.25 * spread, d);
            float arm = 0.5 * bm + 0.354 * s;       // the Kelvin angle, tan 19.47 deg
            float arms = exp(-s / (2.5 * L)) * smoothstep(1.2 + 0.03 * s, 0.0, abs(d - arm));
            w = max(w, max(trail, 0.6 * arms));
            slope += sd * sign(side) * sin(6.2831853 * (d - arm) / 3.0) * arms * 0.12 * k;
            slope += vec2(wa_noised(p * 0.9 + f * t).yz) * trail * 0.08 * k;
        }
        float fn = wa_noise(p * 1.6 - f * t * 0.3) * 0.6 + wa_noise(p * 4.5 + vec2(t * 0.4, 0.0)) * 0.4;
        foam = max(foam, clamp(w * k * (0.2 + 1.1 * fn) - 0.25, 0.0, 1.0));
    }
}

float wa_lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// The water's colour (linear RGB) at world position wpos, seen from campos.
// vel: the current (m/s, world xz); shore: metres to the nearest dry ground; sea: 0/1;
// wind = (dir x, dir z, speed m/s, -); surf = (front speed m/s, period s);
// sun_l: toward the sun; sun_c: its colour.
vec3 water_colour(vec2 uv, vec3 wpos, vec3 campos, vec2 vel, float shore, float sea, float t,
                  vec4 wind, vec2 surf, mat4 boatA, mat4 boatB, vec3 sun_l, vec3 sun_c) {
    vec2 p = wpos.xz;
    vec3 v = normalize(campos - wpos);
    float dist = length(campos - wpos);
    float speed = length(vel);
    vec2 dir = speed > 1e-4 ? vel / speed : vec2(0.0, 1.0);

    // A river's surface moves with its current, the wind's ripples riding on it smaller; on
    // still water (a lake) the wind's ripples are all there is.
    float river = sea > 0.5 ? 0.0 : smoothstep(0.05, 0.3, speed);
    vec2 slope = sea > 0.5 ? wa_current(p, vel, t) * 0.6 + wa_wind(p, vel, wind.xy, wind.z, t)
                           : wa_current(p, vel, t) * mix(0.3, 1.5, river) + wa_wind(p, vel, wind.xy, wind.z, t) * mix(1.0, 0.4, river);
    float foam = 0.0;
    float hp_keep = sea > 0.5 ? 0.7 : 1.0;          // the photograph's fine ripples
    float retime = 1.0;
    if (sea > 0.5) {
        // Where the photograph shows white water: brighter than the water around it.
        float pf = smoothstep(0.03, 0.14, wa_lum(water_tex(uv, 3.0)) - wa_lum(water_tex(uv, 6.5)));
        float lam = max(surf.x * surf.y, 4.0);
        // Each stretch of shore on its own phase, so the fronts arrive in sets, not as one ring.
        float ph = 6.2831853 * (shore / lam + t / max(surf.y, 0.5) + 0.45 * wa_noise(p * 0.015));
        float zone = smoothstep(45.0, 6.0, shore);
        slope += -dir * (6.2831853 / lam) * 0.35 * cos(ph) * (0.4 + 0.6 * zone);
        float crest = pow(0.5 + 0.5 * sin(ph), 5.0);
        float fn = wa_noise(p * 0.3 - dir * t * surf.x * 0.3) * 0.55 + wa_noise(p * 1.2 + dir * t * 0.5) * 0.45;
        // Broken water runs in with each front where the photograph has surf (and in the
        // last few metres anywhere); elsewhere the swell only lifts the surface.
        float breaks = max(pf, smoothstep(14.0, 3.0, shore));
        float broken = smoothstep(0.6, 0.92, crest * 0.85 + fn * 0.45) * smoothstep(28.0, 5.0, shore) * breaks;
        // The swash: a foam sheet at the waterline whose seaward edge runs out as each
        // front arrives and drains back.
        float run = 0.5 + 0.5 * sin(ph - 6.2831853 * shore / lam - 0.9);
        float edge = mix(1.5, 7.0, run);
        float swash = smoothstep(edge, edge - 2.5, shore) * (0.45 + 0.55 * fn) * (0.5 + 0.5 * run) * (0.5 + 0.5 * breaks);
        foam = max(broken * 0.8, swash * 0.75);
        // The photographed surf is re-timed: it shows as a front passes and fades between.
        hp_keep = mix(0.7, 0.3 + 1.0 * crest, zone * pf);
        retime = 1.0 + 0.35 * pf * zone * (crest - 0.35);
    }
    wa_wakes(p, t, boatA, boatB, slope, foam);

    // uv per metre of the surface here (screen derivatives).
    mat2 DU = mat2(dFdx(uv), dFdy(uv));
    mat2 DP = mat2(dFdx(p), dFdy(p));
    float det = determinant(DP);
    mat2 J = abs(det) > 1e-10 ? DU * inverse(DP) : mat2(0.0);
    float lim = 0.004;
    vec2 duv = clamp(J * (slope * 0.35), vec2(-lim), vec2(lim));

    // The photograph's fine ripples ride the current: carried along it in two copies half a
    // cycle apart (each spot on its own clock), the contrast held; its broad shapes -- the
    // reflections, the colour, the photographed foam -- stay where they are.
    float T = 6.0;
    float j = wa_noise(p * 0.05 + 3.1);
    float f0 = fract(t / T + j), f1 = fract(t / T + j + 0.5);
    vec2 uvel = J * vel;
    vec2 o0 = clamp(uvel * (f0 - 0.5) * T, vec2(-0.01), vec2(0.01));
    vec2 o1 = clamp(uvel * (f1 - 0.5) * T, vec2(-0.01), vec2(0.01));
    vec3 low = water_tex(uv + duv, 3.0);
    vec3 h0 = water_tex(uv + duv - o0, 0.0) - water_tex(uv + duv - o0, 3.0);
    vec3 h1 = water_tex(uv + duv - o1, 0.0) - water_tex(uv + duv - o1, 3.0);
    float wb = 1.0 - abs(2.0 * f0 - 1.0);
    vec3 hp = (h0 * wb + h1 * (1.0 - wb)) / sqrt(wb * wb + (1.0 - wb) * (1.0 - wb));
    vec3 photo = low + hp;
    vec3 base = (low + hp * hp_keep) * retime;
    float grazing = 1.0 - clamp(v.y, 0.0, 1.0);
    vec2 vxz = normalize(v.xz + vec2(1e-6));
    float tilt = dot(slope, vxz);
    base *= clamp(1.0 - tilt * (0.8 + 1.2 * grazing), 0.6, 1.5);
    vec3 n = normalize(vec3(-slope.x, 1.0, -slope.y));
    vec3 r = reflect(-v, n);
    float glint = pow(max(dot(r, sun_l), 0.0), 350.0) * smoothstep(0.25, 0.75, wa_lum(low)) * smoothstep(400.0, 30.0, dist);
    base += sun_c * glint * 1.2;
    vec3 foam_c = clamp(mix(vec3(wa_lum(low)), low, 0.35) * 2.2 + 0.08, 0.0, 1.0);
    foam_c = max(foam_c, photo);
    return mix(base, foam_c, clamp(foam, 0.0, 0.92));
}
