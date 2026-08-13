"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { Tier } from "@/lib/holo/engine";

export interface ScenePoint {
  id: string;
  point3d: [number, number, number];
  tier: Tier;
  weight: number;
  isPeak: boolean;
  peakStrength: number;
}

interface Props {
  points: ScenePoint[];
  fieldMagnitude: number;
  queryPoint: [number, number, number] | null;
}

const TIER_COLOR: Record<Tier, number> = {
  recent: 0x5eb4ff,
  repeated: 0xffb24d,
  "long-term": 0xb583ff,
};

/** Small radial-gradient sprite texture used for soft point glows. */
function makeGlowTexture(): THREE.Texture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.5)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

function makeStarfield(glow: THREE.Texture): THREE.Points {
  const count = 1100;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = 6 + Math.random() * 16;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    size: 0.045,
    map: glow,
    transparent: true,
    opacity: 0.3,
    color: 0x9fb8ff,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  return new THREE.Points(geo, mat);
}

/** View-dependent rim glow ("atmosphere") shell wrapped around the sphere. */
function makeAtmosphere(): THREE.Mesh {
  const material = new THREE.ShaderMaterial({
    uniforms: { glowColor: { value: new THREE.Color(0x5ea8ff) } },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vPositionNormal;
      void main() {
        vNormal = normalize( normalMatrix * normal );
        vPositionNormal = normalize( ( modelViewMatrix * vec4(position, 1.0) ).xyz );
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vPositionNormal;
      uniform vec3 glowColor;
      void main() {
        float intensity = pow(0.6 - dot(vNormal, vPositionNormal), 3.0);
        gl_FragColor = vec4(glowColor, clamp(intensity, 0.0, 0.5));
      }
    `,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  return new THREE.Mesh(new THREE.SphereGeometry(1.12, 48, 32), material);
}

interface EnergyStrand {
  curve: THREE.QuadraticBezierCurve3;
  packet: THREE.Sprite;
  speed: number;
  offset: number;
}

export default function SphereScene({ points, fieldMagnitude, queryPoint }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    composer: EffectComposer;
    bloom: UnrealBloomPass;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    group: THREE.Group;
    field: THREE.Mesh;
    fieldGlow: THREE.Sprite;
    lines: THREE.Group;
    energyGroup: THREE.Group;
    energyStrands: EnergyStrand[];
    queryMarker: THREE.Mesh;
    glowTex: THREE.Texture;
    raf: number;
  } | null>(null);

  // one-time scene setup
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05050c);
    scene.fog = new THREE.FogExp2(0x05050c, 0.05);

    const camera = new THREE.PerspectiveCamera(
      45,
      mount.clientWidth / mount.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 0.7, 3.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.85;
    mount.appendChild(renderer.domElement);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
      new THREE.Vector2(mount.clientWidth, mount.clientHeight),
      0.4, // strength
      0.4, // radius
      0.45 // threshold
    );
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 1.8;
    controls.maxDistance = 8;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;

    const hemi = new THREE.HemisphereLight(0x8fb8ff, 0x0a0a18, 0.6);
    const point = new THREE.PointLight(0x66ccff, 1.4, 20);
    point.position.set(2.5, 2, 2.5);
    const point2 = new THREE.PointLight(0xff88cc, 0.6, 20);
    point2.position.set(-2.5, -1.5, -2);
    scene.add(hemi, point, point2);

    const group = new THREE.Group();
    scene.add(group);

    const glowTex = makeGlowTexture();
    scene.add(makeStarfield(glowTex));
    group.add(makeAtmosphere());

    const wireSphere = new THREE.Mesh(
      new THREE.SphereGeometry(1, 40, 28),
      new THREE.MeshBasicMaterial({
        color: 0x3a5fb0,
        wireframe: true,
        transparent: true,
        opacity: 0.13,
      })
    );
    group.add(wireSphere);

    const field = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.35, 4),
      new THREE.MeshStandardMaterial({
        color: 0x8ad4ff,
        emissive: 0x2f6ea8,
        emissiveIntensity: 0.6,
        roughness: 0.15,
        metalness: 0.55,
        transparent: true,
        opacity: 0.92,
      })
    );
    group.add(field);

    const fieldGlow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: glowTex,
        color: 0x7fd0ff,
        transparent: true,
        opacity: 0.3,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    fieldGlow.scale.setScalar(1.1);
    group.add(fieldGlow);

    const lines = new THREE.Group();
    group.add(lines);
    const energyGroup = new THREE.Group();
    group.add(energyGroup);

    const queryMarker = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.05),
      new THREE.MeshStandardMaterial({
        color: 0xff5566,
        emissive: 0xcc2244,
        emissiveIntensity: 0.9,
      })
    );
    queryMarker.visible = false;
    group.add(queryMarker);

    const onResize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      composer.setSize(mount.clientWidth, mount.clientHeight);
      bloom.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);

    const animate = () => {
      controls.update();
      const t = Date.now() * 0.001;
      field.rotation.y += 0.01;
      field.rotation.x += 0.004;
      const pulse = 1 + Math.sin(t * 1.6) * 0.04;
      fieldGlow.scale.setScalar(1.1 * pulse);
      queryMarker.rotation.y += 0.05;
      queryMarker.rotation.x += 0.03;

      const s = stateRef.current;
      if (s) {
        for (const strand of s.energyStrands) {
          const u = (t * strand.speed + strand.offset) % 1;
          strand.packet.position.copy(strand.curve.getPoint(u));
          const mat = strand.packet.material as THREE.SpriteMaterial;
          mat.opacity = 0.25 + 0.55 * Math.sin(u * Math.PI); // fade in/out along the path
        }
      }

      composer.render();
      raf = requestAnimationFrame(animate);
    };
    let raf = requestAnimationFrame(animate);

    stateRef.current = {
      renderer,
      composer,
      bloom,
      scene,
      camera,
      controls,
      group,
      field,
      fieldGlow,
      lines,
      energyGroup,
      energyStrands: [],
      queryMarker,
      glowTex,
      raf,
    };

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      composer.dispose();
      renderer.dispose();
      glowTex.dispose();
      mount.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, []);

  // sync points / field / query into the scene whenever data changes
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;

    const toRemove = s.group.children.filter((c) => c.userData.isMemoryPoint);
    for (const c of toRemove) s.group.remove(c);
    while (s.lines.children.length) s.lines.remove(s.lines.children[0]);
    while (s.energyGroup.children.length) s.energyGroup.remove(s.energyGroup.children[0]);
    s.energyStrands = [];

    for (const p of points) {
      const size = p.isPeak ? 0.06 + 0.05 * p.peakStrength : 0.028 + 0.018 * Math.min(p.weight, 3);
      const geo = new THREE.SphereGeometry(size, 20, 14);
      const mat = new THREE.MeshPhysicalMaterial({
        color: TIER_COLOR[p.tier],
        emissive: p.isPeak ? 0xffffff : TIER_COLOR[p.tier],
        emissiveIntensity: p.isPeak ? 0.6 : 0.12,
        roughness: 0.25,
        metalness: 0.2,
        clearcoat: 0.6,
        clearcoatRoughness: 0.3,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...p.point3d);
      mesh.userData.isMemoryPoint = true;
      s.group.add(mesh);

      if (p.isPeak) {
        const glow = new THREE.Sprite(
          new THREE.SpriteMaterial({
            map: s.glowTex,
            color: TIER_COLOR[p.tier],
            transparent: true,
            opacity: 0.28 + 0.25 * p.peakStrength,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
          })
        );
        glow.scale.setScalar(0.18 + 0.2 * p.peakStrength);
        glow.position.set(...p.point3d);
        glow.userData.isMemoryPoint = true;
        s.group.add(glow);

        if (queryPoint) {
          const from = new THREE.Vector3(...queryPoint);
          const to = new THREE.Vector3(...p.point3d);
          const mid = from.clone().add(to).multiplyScalar(0.5).normalize().multiplyScalar(1.28);
          const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
          const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
          const lineMat = new THREE.LineBasicMaterial({
            color: 0xffe066,
            transparent: true,
            opacity: 0.22 + 0.4 * p.peakStrength,
          });
          s.lines.add(new THREE.Line(lineGeo, lineMat));

          const packet = new THREE.Sprite(
            new THREE.SpriteMaterial({
              map: s.glowTex,
              color: 0xffe066,
              transparent: true,
              opacity: 0.7,
              depthWrite: false,
              blending: THREE.AdditiveBlending,
            })
          );
          packet.scale.setScalar(0.09 + 0.06 * p.peakStrength);
          s.energyGroup.add(packet);
          s.energyStrands.push({ curve, packet, speed: 0.35 + 0.25 * p.peakStrength, offset: Math.random() });
        }
      }
    }

    if (queryPoint) {
      s.queryMarker.visible = true;
      s.queryMarker.position.set(...queryPoint);
    } else {
      s.queryMarker.visible = false;
    }

    const scale = 0.3 + Math.min(fieldMagnitude, 4) * 0.18;
    s.field.scale.setScalar(scale);
  }, [points, fieldMagnitude, queryPoint]);

  return <div ref={mountRef} className="h-full w-full cursor-grab active:cursor-grabbing" />;
}
