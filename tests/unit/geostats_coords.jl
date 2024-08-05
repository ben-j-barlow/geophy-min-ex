# TEST - check geostats coordinates match observation coordinates


# run gen : inspect s0 and a vs copied version
out = gen(m, s0, a, m.rng);
o = out[:o];
s1 = out[:sp];
r = out[:r];

o.geophysical_obs.reading # 0.2630107748053761
o.geophysical_obs.base_map_coordinates # 10, 18

#o_copy = deepcopy(o);
#s1_copy = deepcopy(s1);
#println("Mismatched properties: " * join([p for p in propertynames(s0) if !isequal(getproperty(s0, p), getproperty(s0_copy, p))], ", "))
# I HAVE IMPLEMENTED THE ISEQUAL FUNCTION FOR GeophysicalObservations BUT I HAVE NOT TESTED IT
#x, y = o.agent_pos_x, o.agent_pos_y;
#@assert last(s1.agent_pos_x) == x
#@assert last(s1.agent_pos_y) == y

# run update : inspect b0, a, o vs copied version
b1, ui = update_info(up, b0, a, o);

# timestep 2
out = gen(m, s1, a, m.rng);
o = out[:o];
s2 = out[:sp];
b2, ui = update_info(up, b1, a, o);

o.geophysical_obs.reading # 0.29380504570879085
o.geophysical_obs.base_map_coordinates # 11, 19
b2.geostats.geophysical_data.reading
b2.geostats.geophysical_data.base_map_coordinates


# timestep 3
out = gen(m, s2, a, m.rng);
o = out[:o];
s3 = out[:sp];
b3, ui = update_info(up, b2, a, o);

o.geophysical_obs.reading # 0.323766718336902
o.geophysical_obs.base_map_coordinates # 12, 20

b3.geostats.geophysical_data.reading
b3.geostats.geophysical_data.base_map_coordinates

s = s3;
b = b3;

for i in 1:5
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);
    
    @info ""
    @info "timestep $i"
    @info "b geostats base map coords  $(b.geostats.geophysical_data.base_map_coordinates)"
    @info "s normalized agent position x $(last(s.agent_pos_x) / m.base_grid_element_length) y $(last(s.agent_pos_y) / m.base_grid_element_length)"
    @info "obs base map coords         $(o.geophysical_obs.base_map_coordinates)"
    @info "bp geostats base map coords $(bp.geostats.geophysical_data.base_map_coordinates)"

    b = bp
    s = sp
end

plot(b, m, s)

plot(b3, m, s3)

b.geostats.geophysical_data.base_map_coordinates

