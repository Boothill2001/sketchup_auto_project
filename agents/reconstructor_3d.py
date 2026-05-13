"""
Agent 7 — 3D Reconstructor (NEW)
Phase 5b: Convert architectural elements (walls, slabs, doors, windows, stairs)
into SketchUp Ruby script geometry.

This agent runs IN PARALLEL with the Coder (steel members).
Together they produce a complete LOD300 model with BOTH structural steel AND
architectural elements — walls, floors, openings, stairs.

Architecture:
  - Walls → groups of rectangular faces extruded to height
  - Slabs  → horizontal planes at each level
  - Doors/windows → openings + simple frame geometry
  - Stairs → extruded steps

Combined Ruby output merges into lod300_model.rb alongside steel members.
"""

import json
from pathlib import Path
from rich import print as rprint

from config import OUTPUT_JSON_DIR, RUBY_OUTPUT_DIR
from core.analysis_context import load_analysis_dict, build_coder_context
from core.llm_wrapper import call_llm

ARCHITECTURAL_OUTPUT_FILE = str(Path(OUTPUT_JSON_DIR) / "architectural_elements.json")
ARCH_RUBY_OUTPUT_FILE = str(Path(RUBY_OUTPUT_DIR) / "lod300_architectural.rb")


WALL_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# WALLS — Generated from architectural plan extraction
# ──────────────────────────────────────────────────────────────
def create_wall(start_pt, end_pt, height_mm, thickness_mm, wall_name, material_name="Wall_Concrete")
  # Create wall as a rectangular face extruded vertically
  width = Math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
  angle = Math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0]) * 180.0 / Math::PI
  
  # Center point
  cx = (start_pt[0] + end_pt[0]) / 2.0
  cy = (start_pt[1] + end_pt[1]) / 2.0
  
  # Create wall group
  wall_group = Sketchup.active_model.entities.add_group
  wall_group.name = wall_name
  
  entities = wall_group.entities
  
  # Create wall profile face at origin
  pts = [
    [-width/2, -thickness_mm/2, 0],
    [ width/2, -thickness_mm/2, 0],
    [ width/2,  thickness_mm/2, 0],
    [-width/2,  thickness_mm/2, 0]
  ]
  face = entities.add_face(pts)
  
  # Extrude upward
  face.pushpull(height_mm)
  
  # Transform: rotate and position
  tr = Geom::Transformation.new(
    Geom::Point3d.new(cx, cy, 0),
    Z_AXIS,
    angle
  )
  wall_group.transform!(tr)
  
  # Assign material
  begin
    mat = Sketchup.active_model.materials[material_name]
    unless mat
      mat = Sketchup.active_model.materials.add(material_name)
      mat.color = Sketchup::Color.new(200, 200, 200)  # light gray
    end
    wall_group.material = mat
  rescue
  end
  
  # Add to layer
  layer_name = "LOD300_Walls"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  wall_group.layer = layer
  
  wall_group
end

# ── Wall instances ──────────────────────────────────────────
{wall_calls}
'''


SLAB_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# SLABS / FLOORS — Generated from architectural plan extraction
# ──────────────────────────────────────────────────────────────
def create_slab(boundary_pts, z_mm, thickness_mm, slab_name, material_name="Floor_Concrete")
  slab_group = Sketchup.active_model.entities.add_group
  slab_group.name = slab_name
  
  entities = slab_group.entities
  
  # Create face from boundary points
  pts_3d = boundary_pts.map {{ |p| [p[0], p[1], z_mm] }}
  face = entities.add_face(pts_3d)
  
  # Extrude downward
  face.reverse! unless face.normal.samedirection?(Z_AXIS)
  face.pushpull(-thickness_mm)
  
  # Assign material
  begin
    mat = Sketchup.active_model.materials[material_name]
    unless mat
      mat = Sketchup.active_model.materials.add(material_name)
      mat.color = Sketchup::Color.new(180, 190, 200)
    end
    slab_group.material = mat
  rescue
  end
  
  layer_name = "LOD300_Slabs"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  slab_group.layer = layer
  
  slab_group
end

# ── Slab instances ──────────────────────────────────────────
{slab_calls}
'''


DOOR_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# DOORS — Generated from architectural plan extraction
# ──────────────────────────────────────────────────────────────
def create_door(x, y, z, width_mm, height_mm, door_name, material_name="Door_Frame")
  door_group = Sketchup.active_model.entities.add_group
  door_group.name = door_name
  
  entities = door_group.entities
  
  # Simple door panel
  pts = [
    [x - width_mm/2, y - 25, z],
    [x + width_mm/2, y - 25, z],
    [x + width_mm/2, y + 25, z],
    [x - width_mm/2, y + 25, z]
  ]
  face = entities.add_face(pts)
  face.pushpull(height_mm)
  
  begin
    mat = Sketchup.active_model.materials[material_name]
    unless mat
      mat = Sketchup.active_model.materials.add(material_name)
      mat.color = Sketchup::Color.new(139, 90, 43)  # brown
    end
    door_group.material = mat
  rescue
  end
  
  layer_name = "LOD300_Doors"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  door_group.layer = layer
  
  door_group
end

# ── Door instances ──────────────────────────────────────────
{door_calls}
'''


WINDOW_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# WINDOWS — Generated from architectural plan extraction
# ──────────────────────────────────────────────────────────────
def create_window(x, y, z_sill, width_mm, height_mm, window_name, material_name="Window_Glass")
  window_group = Sketchup.active_model.entities.add_group
  window_group.name = window_name
  
  entities = window_group.entities
  
  # Simple window panel
  pts = [
    [x - width_mm/2, y - 25, z_sill],
    [x + width_mm/2, y - 25, z_sill],
    [x + width_mm/2, y + 25, z_sill],
    [x - width_mm/2, y + 25, z_sill]
  ]
  face = entities.add_face(pts)
  face.pushpull(height_mm)
  
  begin
    mat = Sketchup.active_model.materials[material_name]
    unless mat
      mat = Sketchup.active_model.materials.add(material_name)
      mat.color = Sketchup::Color.new(135, 206, 235)  # light blue
      mat.alpha = 0.5
    end
    window_group.material = mat
  rescue
  end
  
  layer_name = "LOD300_Windows"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  window_group.layer = layer
  
  window_group
end

# ── Window instances ─────────────────────────────────────────
{window_calls}
'''


STAIR_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# STAIRS — Generated from architectural plan extraction
# ──────────────────────────────────────────────────────────────
def create_stairs(x, y, z_start, width_mm, rise_total_mm, run_total_mm, num_risers, stair_name)
  stair_group = Sketchup.active_model.entities.add_group
  stair_group.name = stair_name
  
  entities = stair_group.entities
  
  rise_per_step = rise_total_mm / num_risers.to_f
  tread_depth = run_total_mm / (num_risers - 1).to_f
  
  (0...num_risers).each do |i|
    # Horizontal tread
    z = z_start + (i * rise_per_step)
    y_step = y + (i * tread_depth)
    pts = [
      [x,             y_step,             z],
      [x + width_mm,  y_step,             z],
      [x + width_mm,  y_step + tread_depth, z],
      [x,             y_step + tread_depth, z]
    ]
    entities.add_face(pts).pushpull(rise_per_step) rescue nil
  end
  
  layer_name = "LOD300_Stairs"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  stair_group.layer = layer
  
  stair_group
end

# ── Stair instances ─────────────────────────────────────────
{stair_calls}
'''


COLUMN_ARCH_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# ARCHITECTURAL COLUMNS — from plan extraction
# ──────────────────────────────────────────────────────────────
def create_arch_column(x, y, z_base, width_mm, depth_mm, height_mm, col_name)
  col_group = Sketchup.active_model.entities.add_group
  col_group.name = col_name
  
  entities = col_group.entities
  
  pts = [
    [x - width_mm/2, y - depth_mm/2, z_base],
    [x + width_mm/2, y - depth_mm/2, z_base],
    [x + width_mm/2, y + depth_mm/2, z_base],
    [x - width_mm/2, y + depth_mm/2, z_base]
  ]
  face = entities.add_face(pts)
  face.pushpull(height_mm)
  
  layer_name = "LOD300_Columns"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  col_group.layer = layer
  
  col_group
end

# ── Architectural Column instances ───────────────────────────
{col_calls}
'''


ROOF_RUBY_TEMPLATE = '''
# ──────────────────────────────────────────────────────────────
# ROOF — Generated from architectural elevation extraction
# ──────────────────────────────────────────────────────────────
def create_roof(boundary_pts, z_eaves, slope_degrees, overhang_mm, roof_name)
  roof_group = Sketchup.active_model.entities.add_group
  roof_group.name = roof_name
  
  entities = roof_group.entities
  
  # Create roof plane(s) based on boundary
  # For now: simple flat roof with overhang
  pts_3d = boundary_pts.map {{ |p| [p[0], p[1], z_eaves] }}
  face = entities.add_face(pts_3d)
  
  layer_name = "LOD300_Roof"
  layer = Sketchup.active_model.layers[layer_name]
  unless layer
    layer = Sketchup.active_model.layers.add(layer_name)
  end
  roof_group.layer = layer
  
  roof_group
end

# ── Roof instance ───────────────────────────────────────────
{roof_calls}
'''


# ── HEADER (constants and setup) ────────────────────────────────
HEADER = '''# ╔══════════════════════════════════════════════════════════════╗
# ║  LOD 300 — ARCHITECTURAL ELEMENTS                             ║
# ║  Auto-generated from PDF plan + elevation extraction          ║
# ╚══════════════════════════════════════════════════════════════╝
# 
# Load into SketchUp:  Extensions → Ruby Console
#                      load 'path/to/lod300_architectural.rb'

Z_AXIS  = Geom::Vector3d.new(0, 0, 1)
X_AXIS  = Geom::Vector3d.new(1, 0, 0)

# Convert mm to SketchUp inches (SketchUp internal unit = inch)
MM_TO_INCH = 1.0 / 25.4

def mm_to_inch(val)
  val * MM_TO_INCH
end

# Create layer structure
LAYERS = [
  "LOD300_Walls",
  "LOD300_Slabs",
  "LOD300_Doors",
  "LOD300_Windows",
  "LOD300_Stairs",
  "LOD300_Columns",
  "LOD300_Roof"
]
LAYERS.each do |layer_name|
  unless Sketchup.active_model.layers[layer_name]
    Sketchup.active_model.layers.add(layer_name)
  end
end

puts "LOD300 Architectural layers created."
'''


def build_architectural_ruby(
    arch_elements: dict,
    pdf_analysis: dict = None,
) -> str:
    """
    Generate Ruby script for all architectural elements.
    Args:
        arch_elements: dict from architectural_extractor.py
        pdf_analysis: Phase 0 convention analysis
    Returns:
        Complete Ruby script string
    """
    if pdf_analysis is None:
        pdf_analysis = load_analysis_dict()

    walls = arch_elements.get("walls", [])
    doors = arch_elements.get("doors", [])
    windows = arch_elements.get("windows", [])
    stairs = arch_elements.get("stairs", [])
    slabs = arch_elements.get("slabs", [])
    columns = arch_elements.get("columns", [])
    roof = arch_elements.get("roof", {})

    # ── Generate wall Ruby code ────────────────────────────────
    wall_lines = []
    for i, w in enumerate(walls):
        sp = w.get("start_point", {})
        ep = w.get("end_point", {})
        h = w.get("height_mm", 3500)
        t = w.get("thickness_mm", 200)
        name = w.get("mark", f"Wall_{i+1}")
        sx, sy = sp.get("x", 0), sp.get("y", 0)
        ex, ey = ep.get("x", 1000), ep.get("y", 0)
        wall_lines.append(
            f'  create_wall([{sx},{sy}], [{ex},{ey}], {h}, {t}, "{name}")'
        )
    wall_code = WALL_RUBY_TEMPLATE.replace("{wall_calls}", "\n".join(wall_lines) or "  # No walls extracted")

    # ── Generate slab Ruby code ─────────────────────────────────
    slab_lines = []
    for i, s in enumerate(slabs):
        z = s.get("z_mm", 0)
        t = s.get("thickness_mm", 200)
        name = s.get("level_name", f"Slab_{i+1}")
        boundary = s.get("boundary_points", [[0, 0], [10000, 0], [10000, 10000], [0, 10000]])
        pts_formatted = ", ".join(f"[{p[0]},{p[1]}]" for p in boundary[:20])
        slab_lines.append(f'  create_slab([{pts_formatted}], {z}, {t}, "{name}")')
    slab_code = SLAB_RUBY_TEMPLATE.replace("{slab_calls}", "\n".join(slab_lines) or "  # No slabs extracted")

    # ── Generate door Ruby code ─────────────────────────────────
    door_lines = []
    for d in doors:
        pos = d.get("position", {})
        x, y = pos.get("x", 0), pos.get("y", 0)
        w = d.get("width_mm", 900)
        h = d.get("height_mm", 2100)
        name = d.get("mark", f"Door_{len(door_lines)+1}")
        door_lines.append(f'  create_door({x}, {y}, 0, {w}, {h}, "{name}")')
    door_code = DOOR_RUBY_TEMPLATE.replace("{door_calls}", "\n".join(door_lines) or "  # No doors extracted")

    # ── Generate window Ruby code ───────────────────────────────
    window_lines = []
    for wi in windows:
        pos = wi.get("position", {})
        x, y = pos.get("x", 0), pos.get("y", 0)
        s_h = wi.get("sill_height_mm", 900)
        w_w = wi.get("width_mm", 1200)
        w_h = wi.get("height_mm", 1200)
        name = wi.get("mark", f"Window_{len(window_lines)+1}")
        window_lines.append(f'  create_window({x}, {y}, {s_h}, {w_w}, {w_h}, "{name}")')
    window_code = WINDOW_RUBY_TEMPLATE.replace("{window_calls}", "\n".join(window_lines) or "  # No windows extracted")

    # ── Generate stair Ruby code ─────────────────────────────────
    stair_lines = []
    for st in stairs:
        pos = st.get("position", {})
        x, y = pos.get("x", 0), pos.get("y", 0)
        w = st.get("width_mm", 1000)
        rise = st.get("rise_mm", 3500)
        run = st.get("run_mm", 3000)
        n = st.get("num_risers", 20)
        name = st.get("mark", f"Stair_{len(stair_lines)+1}")
        stair_lines.append(f'  create_stairs({x}, {y}, 0, {w}, {rise}, {run}, {n}, "{name}")')
    stair_code = STAIR_RUBY_TEMPLATE.replace("{stair_calls}", "\n".join(stair_lines) or "  # No stairs extracted")

    # ── Generate architectural column Ruby code ──────────────────
    col_lines = []
    for i, c in enumerate(columns):
        pos = c.get("position", {})
        x, y = pos.get("x", 0), pos.get("y", 0)
        w = c.get("width_mm", 400)
        d = c.get("depth_mm", 400)
        h = c.get("height_mm", 3500)
        name = c.get("mark", f"ArchCol_{i+1}")
        col_lines.append(f'  create_arch_column({x}, {y}, 0, {w}, {d}, {h}, "{name}")')
    col_code = COLUMN_ARCH_RUBY_TEMPLATE.replace("{col_calls}", "\n".join(col_lines) or "  # No arch columns")

    # ── Generate roof Ruby code ──────────────────────────────────
    roof_lines = []
    if roof:
        z_eaves = roof.get("eaves_z_mm", 3500)
        slope = roof.get("slope_degrees", 0)
        building = arch_elements.get("building_outline", {})
        bw = building.get("overall_width_mm", 10000)
        bd = building.get("overall_depth_mm", 10000)
        if bw and bd:
            boundary = [[0, 0], [bw, 0], [bw, bd], [0, bd]]
            pts_formatted = ", ".join(f"[{p[0]},{p[1]}]" for p in boundary)
            roof_lines.append(f'  create_roof([{pts_formatted}], {z_eaves}, {slope}, 500, "Roof")')
    roof_code = ROOF_RUBY_TEMPLATE.replace("{roof_calls}", "\n".join(roof_lines) or "  # No roof extracted")

    # ── Assemble full script ────────────────────────────────────
    sections = [
        HEADER,
        wall_code,
        slab_code,
        col_code,
        door_code,
        window_code,
        stair_code,
        roof_code,
        "\nputs 'LOD300 Architectural elements loaded successfully.'",
    ]

    script = "\n\n".join(sections)
    return script


def save_architectural_ruby(script: str, output_path: str = None) -> str:
    """Save architectural Ruby script to file."""
    if output_path is None:
        output_path = ARCH_RUBY_OUTPUT_FILE
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(script, encoding="utf-8")
    rprint(f"[green]Architectural Ruby script saved:[/] {output_path}")
    return output_path


def load_architectural_elements() -> dict:
    """Load previously extracted architectural elements from JSON."""
    path = Path(ARCHITECTURAL_OUTPUT_FILE)
    if not path.exists():
        rprint("[yellow]architectural_elements.json not found — returning empty.[/]")
        return _empty_elements()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_elements()


def _empty_elements() -> dict:
    return {
        "walls": [],
        "doors": [],
        "windows": [],
        "stairs": [],
        "slabs": [],
        "columns": [],
        "grid_lines": {"x_axis": [], "y_axis": [], "spacing_mm": 6000},
        "rooms": [],
        "floor_levels": [],
        "roof": {},
        "building_outline": {},
        "floor_level": "Floor 1",
        "floor_to_floor_height_mm": 3500,
    }


def merge_ruby_scripts(steel_rb_path: str, arch_rb_path: str, output_path: str) -> str:
    """
    Merge structural steel Ruby script with architectural Ruby script
    into a single lod300_model.rb.
    """
    steel_rb = Path(steel_rb_path).read_text(encoding="utf-8") if Path(steel_rb_path).exists() else ""
    arch_rb = Path(arch_rb_path).read_text(encoding="utf-8") if Path(arch_rb_path).exists() else ""

    # Add merge header
    merge_header = """# ╔══════════════════════════════════════════════════════════════╗
# ║  LOD 300 — COMPLETE MODEL (Structural + Architectural)        ║
# ║  Merged from lod300_model.rb + lod300_architectural.rb        ║
# ╚══════════════════════════════════════════════════════════════╝

"""

    merged = merge_header + steel_rb + "\n\n" + arch_rb

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(merged, encoding="utf-8")

    rprint(f"[bold green]Merged Ruby script:[/] {output_path}")
    return merged