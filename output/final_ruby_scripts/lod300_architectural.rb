# ╔══════════════════════════════════════════════════════════════╗
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

puts 'No architectural elements extracted — steel-only model.'