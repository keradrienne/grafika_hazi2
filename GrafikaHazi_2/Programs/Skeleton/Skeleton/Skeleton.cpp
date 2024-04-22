//=============================================================================================
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kerekes Adrienne
// Neptun : GFVHSO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;	//távolság
	vec3 position, normal;	//hit helye + normál vektora a hitelt felületnek
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;	//szemünk helye + iránya
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct HSzog : public Intersectable {
	vec3 A, B, C;

	HSzog(const vec3& a, const vec3& b, const vec3& c, Material* _material) {
		A = a;
		B = b;
		C = c;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 n = normalize(cross(B - A, C - A));	//felületi normális
		float t = dot(A - ray.start, n) / dot(ray.dir, n);	//távolság
		vec3 p = ray.start + ray.dir * t;	//metszéspont
		if (dot(cross(B - A, p - A), n) >= 0 && dot(cross(C - B, p - B), n) >= 0 && dot(cross(A - C, p - C), n) >= 0) {
			hit.t = t;
			hit.position = p;
			hit.normal = n;
			hit.material = material;
			return hit;
		}
		return hit;
	}
};

struct Kocka : Intersectable
{
	vec3 csucsok[8] = {
		vec3(0, 0, 0),
		vec3(0, 0, 1),
		vec3(0, 1, 0),
		vec3(0, 1, 1),
		vec3(1, 0, 0),
		vec3(1, 0, 1),
		vec3(1, 1, 0),
		vec3(1, 1, 1)
	};



	Kocka(Material* _material) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {

		HSzog hszogek[12] = {
		HSzog(csucsok[0], csucsok[6], csucsok[4], material),
		HSzog(csucsok[0], csucsok[2], csucsok[6], material),
		HSzog(csucsok[0], csucsok[3], csucsok[2], material),
		HSzog(csucsok[0], csucsok[1], csucsok[3], material),
		HSzog(csucsok[2], csucsok[7], csucsok[6], material),
		HSzog(csucsok[2], csucsok[3], csucsok[7], material),
		HSzog(csucsok[4], csucsok[6], csucsok[7], material),
		HSzog(csucsok[4], csucsok[7], csucsok[5], material),
		HSzog(csucsok[0], csucsok[4], csucsok[5], material),
		HSzog(csucsok[0], csucsok[5], csucsok[1], material),
		HSzog(csucsok[1], csucsok[5], csucsok[7], material),
		HSzog(csucsok[1], csucsok[7], csucsok[3], material)
		};

		Hit t1;
		Hit t2;

		for (size_t i = 0; i < 12; i++)
		{
			if (hszogek[i].intersect(ray).t > 0) {
				if (t1.t < 0)
					t1 = hszogek[i].intersect(ray);
				else
					t2 = hszogek[i].intersect(ray);
			}
		}
		return (t1.t > t2.t) ? t1 : t2;
	}
};

struct Ikoza : Intersectable
{
	vec3 csucsok[12] = {
		vec3(0,	-0.525731, 0.850651),
		vec3(0.850651, 0, 0.525731),
		vec3(0.850651, 0, -0.525731),
		vec3(-0.850651, 0, -0.525731),
		vec3(-0.850651, 0, 0.525731),
		vec3(-0.525731,	0.850651, 0),
		vec3(0.525731, 0.850651, 0),
		vec3(0.525731, -0.850651, 0),
		vec3(-0.525731,	-0.850651, 0),
		vec3(0,	-0.525731, -0.850651),
		vec3(0, 0.525731, -0.850651),
		vec3(0, 0.525731, 0.850651)
	};



	Ikoza(Material* _material, const vec3& _scale, const vec3& _eltol) {
		material = _material;

		for (int i = 0; i < 12; i++) {
			vec4 cs = vec4(csucsok[i].x, csucsok[i].y, csucsok[i].z, 1);
			mat4 scale = ScaleMatrix(_scale);
			vec4 scaled_cs = cs * scale;

			csucsok[i] = vec3(scaled_cs.x, scaled_cs.y, scaled_cs.z) + _eltol;

		}
	}

	Hit intersect(const Ray& ray) {
		Hit t1;
		Hit t2;

		HSzog hszogek[20] = {
		HSzog(csucsok[1], csucsok[2], csucsok[6], material),
		HSzog(csucsok[1], csucsok[7], csucsok[2], material),
		HSzog(csucsok[3], csucsok[4], csucsok[5], material),
		HSzog(csucsok[4], csucsok[3], csucsok[8], material),
		HSzog(csucsok[6], csucsok[5], csucsok[11], material),
		HSzog(csucsok[5], csucsok[6], csucsok[10], material),
		HSzog(csucsok[9], csucsok[10], csucsok[2], material),
		HSzog(csucsok[10], csucsok[9], csucsok[3], material),
		HSzog(csucsok[7], csucsok[8], csucsok[9], material),
		HSzog(csucsok[8], csucsok[7], csucsok[0], material),
		HSzog(csucsok[11], csucsok[0], csucsok[1], material),
		HSzog(csucsok[0], csucsok[11], csucsok[4], material),
		HSzog(csucsok[6], csucsok[2], csucsok[10], material),
		HSzog(csucsok[1], csucsok[6], csucsok[11], material),
		HSzog(csucsok[3], csucsok[5], csucsok[10], material),
		HSzog(csucsok[5], csucsok[4], csucsok[11], material),
		HSzog(csucsok[2], csucsok[7], csucsok[9], material),
		HSzog(csucsok[7], csucsok[1], csucsok[0], material),
		HSzog(csucsok[3], csucsok[9], csucsok[8], material),
		HSzog(csucsok[4], csucsok[8], csucsok[0], material)
		};

		for (size_t i = 0; i < 20; i++)
		{
			if (hszogek[i].intersect(ray).t > 0) {
				if (t1.t < 0)
					t1 = hszogek[i].intersect(ray);
				else
					t2 = hszogek[i].intersect(ray);
			}
		}
		return (t1.t > t2.t) ? t2 : t1;
	}
};

struct Dodeka : Intersectable
{
	vec3 csucsok[20] = {
		vec3(-0.57735, -0.57735, 0.57735),
		vec3(0.934172, 0.356822,0),
		vec3(0.934172, -0.356822, 0),
		vec3(-0.934172, 0.356822, 0),
		vec3(-0.934172, -0.356822, 0),
		vec3(0, 0.934172, 0.356822),
		vec3(0, 0.934172, -0.356822),
		vec3(0.356822, 0, -0.934172),
		vec3(-0.356822, 0, -0.934172),
		vec3(0, -0.934172, -0.356822),
		vec3(0, -0.934172, 0.356822),
		vec3(0.356822, 0, 0.934172),
		vec3(-0.356822, 0, 0.934172),
		vec3(0.57735, 0.57735, -0.57735),
		vec3(0.57735, 0.57735, 0.57735),
		vec3(-0.57735, 0.57735, -0.57735),
		vec3(-0.57735 , 0.57735 , 0.57735),
		vec3(0.57735, -0.57735, -0.57735),
		vec3(0.57735, -0.57735, 0.57735),
		vec3(-0.57735, -0.57735, -0.57735)
	};

	Dodeka(Material* _material, const vec3& _scale, const vec3& _eltol) {
		material = _material;

		for (int i = 0; i < 20; i++) {
			vec4 cs = vec4(csucsok[i].x, csucsok[i].y, csucsok[i].z, 1);
			mat4 scale = ScaleMatrix(_scale);
			vec4 scaled_cs = cs * scale;

			csucsok[i] = vec3(scaled_cs.x, scaled_cs.y, scaled_cs.z) + _eltol;

		}
	}

	Hit intersect(const Ray& ray) {

		HSzog hszogek[36] = {
			HSzog(csucsok[18],csucsok[2], csucsok[1], material),
			HSzog(csucsok[11],csucsok[18], csucsok[1], material),
			HSzog(csucsok[14],csucsok[11], csucsok[1], material),
			HSzog(csucsok[7],csucsok[13], csucsok[1], material),
			HSzog(csucsok[17],csucsok[7], csucsok[1], material),
			HSzog(csucsok[2],csucsok[17], csucsok[1], material),
			HSzog(csucsok[19],csucsok[4], csucsok[3], material),
			HSzog(csucsok[8],csucsok[19], csucsok[3], material),
			HSzog(csucsok[15],csucsok[8], csucsok[3], material),
			HSzog(csucsok[12],csucsok[16], csucsok[3], material),
			HSzog(csucsok[0],csucsok[12], csucsok[3], material),
			HSzog(csucsok[4],csucsok[0], csucsok[3], material),
			HSzog(csucsok[6],csucsok[15], csucsok[3], material),
			HSzog(csucsok[5],csucsok[6], csucsok[3], material),
			HSzog(csucsok[16],csucsok[5], csucsok[3], material),
			HSzog(csucsok[5],csucsok[14], csucsok[1], material),
			HSzog(csucsok[6],csucsok[5], csucsok[1], material),
			HSzog(csucsok[13],csucsok[6], csucsok[1], material),
			HSzog(csucsok[9],csucsok[17], csucsok[2], material),
			HSzog(csucsok[10],csucsok[9], csucsok[2], material),
			HSzog(csucsok[18],csucsok[10], csucsok[2], material),
			HSzog(csucsok[10],csucsok[0], csucsok[4], material),
			HSzog(csucsok[9],csucsok[10], csucsok[4], material),
			HSzog(csucsok[19],csucsok[9], csucsok[4], material),
			HSzog(csucsok[19],csucsok[8], csucsok[7], material),
			HSzog(csucsok[9],csucsok[19], csucsok[7], material),
			HSzog(csucsok[17],csucsok[9], csucsok[7], material),
			HSzog(csucsok[8],csucsok[15], csucsok[6], material),
			HSzog(csucsok[7],csucsok[8], csucsok[6], material),
			HSzog(csucsok[13],csucsok[7], csucsok[6], material),
			HSzog(csucsok[11],csucsok[14], csucsok[5], material),
			HSzog(csucsok[12],csucsok[11], csucsok[5], material),
			HSzog(csucsok[16],csucsok[12], csucsok[5], material),
			HSzog(csucsok[12],csucsok[0], csucsok[10], material),
			HSzog(csucsok[11],csucsok[12], csucsok[10], material),
			HSzog(csucsok[18],csucsok[11], csucsok[10], material)
		};

		Hit t1;
		Hit t2;

		for (size_t i = 0; i < 36; i++)
		{
			if (hszogek[i].intersect(ray).t > 0) {
				if (t1.t < 0)
					t1 = hszogek[i].intersect(ray);
				else
					t2 = hszogek[i].intersect(ray);
			}
		}
		return (t1.t > t2.t) ? t2 : t1;
	}
};

struct Kup : Intersectable {
	vec3 p, n;
	float alfa, h;

	Kup(Material* _material, const vec3& _p, const vec3& _n, float _alfa, float _h) {
		material = _material;
		p = _p;
		n = normalize(_n);
		alfa = _alfa;
		h = _h;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t1;
		float t2;
		vec3 dist = ray.start - p;
		float a = powf(dot(ray.dir, n), 2) - dot(ray.dir, ray.dir) * powf(cos(alfa), 2);
		float b = 2 * (dot(ray.dir, n) * dot(dist, n) - dot(ray.dir, dist) * powf(cos(alfa), 2));
		float c = powf(dot(dist, n), 2) - dot(dist, dist) * powf(cos(alfa), 2);
		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);

		if (discr == 0) {
			t1 = -b / (2 * a);
			t2 = -b / (2 * a);
		}
		else if (discr > 0) {
			t1 = (-b + sqrt_discr) / 2.0f / a;
			t2 = (-b - sqrt_discr) / 2.0f / a;
		}
		vec3 hit1 = ray.start + ray.dir * t1;
		vec3 hit2 = ray.start + ray.dir * t2;
		if (t1 < 0 && t2 < 0) return hit;
		if (dot(hit1 - p, n) > 0 && dot(hit1 - p, n) < h) {
			if (dot(hit2 - p, n) > 0 && dot(hit2 - p, n) < h) {
				hit.t = (t1 > t2) ? t2 : t1;
			}
			else hit.t = t1;
		}
		else if (dot(hit2 - p, n) > 0 && dot(hit2 - p, n) < h) {
			hit.t = t2;
		}
		else return hit;

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(2 * dot(hit.position - p, n) * n - 2 * (hit.position - p) * powf(cos(alfa), 2));
		hit.material = material;
		return hit;
	}
};



class Camera {
	vec3 eye, lookat, right, up; //szem + nézet
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		position = _direction;
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

	vec3 K1_p = vec3(0.7f, 1.0f, 0.9f);
	vec3 K1_n = vec3(0.0f, -1.0f, 0.0f);
	vec3 K2_p = vec3(0.0f, 0.8f, 0.3f);
	vec3 K2_n = vec3(1.0f, 0.0f, 0.0f);
	vec3 K3_p = vec3(0.85f, 0.3f, 0.9f);
	vec3 K3_n = vec3(1.0f, 0.0f, -0.5f);
public:
	void build() {
		lights.clear();
		objects.clear();
		vec3 eye = vec3(2.0f, 0.5f, 1.7f), vup = vec3(0.0f, 1.0f, 0.0f), lookat = vec3(0.1f, 0.5f, 0.1f);
		//vec3 eye = vec3(2, 0.5, 1.7), vup = vec3(0, 1, 0), lookat = vec3(0.1, 0.5, 0.1);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.8f, 0.8f, 0.8f);

		vec3 kd(0.2f, 0.2f, 0.2f), ks(2, 2, 2);  //kd szín
		Material* material = new Material(kd, ks, 50);

		//objects.push_back(new HSzog(vec3(-0.3, -0.3, 0), vec3(0.3, -0.3, 0), vec3(0.3, 0.3, 0), material));	
		objects.push_back(new Kocka(material));
		objects.push_back(new Dodeka(material, vec3(0.25, 0.25, 0.25), vec3(0.6, 0.3, 0.9)));
		objects.push_back(new Ikoza(material, vec3(0.18, 0.18, 0.18), vec3(1.1, 0.3, 0.7)));

		Kup* k1 = new Kup(material, K1_p, K1_n, M_PI / 8, 0.1f);
		objects.push_back(k1);
		lights.push_back(new Light(K1_p + k1->n * 0.01f, vec3(1.0f, 0.0f, 0.0f)));

		Kup* k2 = new Kup(material, K2_p, K2_n, M_PI / 8, 0.1);
		objects.push_back(k2);
		lights.push_back(new Light(K2_p + k2->n * 0.01f, vec3(0.0f, 1.0f, 0.0f)));

		Kup* k3 = new Kup(material, K3_p, K3_n, M_PI / 8, 0.1f);
		objects.push_back(k3);
		lights.push_back(new Light(K3_p + k3->n * 0.01f, vec3(0.0f, 0.0f, 1.0f)));

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0.0f, 0.0f, 0.0f);

		vec3 outRadiance = 0.2f * (1.0f + dot(hit.normal, normalize(-ray.dir))) * La;

		for (Light* light : lights) {
			vec3 shadowRayStart = hit.position + hit.normal * epsilon * 1.0f;
			vec3 shadowRayNormal = normalize(light->position - hit.position);
			Ray shadowRay(shadowRayStart, shadowRayNormal);
			Hit shadowHit = firstIntersect(shadowRay);
			float shadow_t = length(shadowHit.position - hit.position);
			float light_t = length(light->position - hit.position);
			if (shadow_t >= light_t) {
				outRadiance = outRadiance + (light->Le * hit.material->kd) / powf(light_t, 2);
			}
		}
		return outRadiance;
	}

	void onClick(int x, int y) {
		Ray clickRay = camera.getRay(x, y);
		Hit hit = firstIntersect(clickRay);

		if (hit.t > 0) {
			float clickKup1_t = length(hit.position - K1_p);
			float clickKup2_t = length(hit.position - K2_p);
			float clickKup3_t = length(hit.position - K3_p);

			float less = clickKup1_t;
			if (less > clickKup2_t)
				less = clickKup2_t;
			else if (less > clickKup3_t)
				less = clickKup3_t;

			if (less == clickKup1_t) {
				K1_p = hit.position;
				K1_n = normalize(hit.normal);
			}
			else if (less == clickKup2_t) {
				K2_p = hit.position;
				K2_n = normalize(hit.normal);
			}
			else {
				K3_p = hit.position;
				K3_n = normalize(hit.normal);
			}

		}
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == 0 && state == 0) {
		scene.onClick(pX, windowHeight - pY);

		scene.build();

		glViewport(0, 0, windowWidth, windowHeight);
		scene.build();

		std::vector<vec4> image(windowWidth * windowHeight);
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		scene.render(image);
		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

		fullScreenTexturedQuad->Draw();
		glutSwapBuffers();
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}

