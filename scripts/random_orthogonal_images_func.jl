function random_orthogonal_images(N::Int) ::Vector{Matrix{Int64}}
    s = Int(sqrt(N))
    
    # All ones
    i1 = reshape(ones(N), s, s)
    
    # Creates half and half
    i2 = copy(i1)
    i2[1 : Int(N/2)] .= 1
    i2[Int(N/2)+1 : N] .= -1
    
    # Random Orthogonal
    i3 = copy(i2)
    
    # Randomly chooses positions to flip, r1 for first half, r2 for second
    r1 = shuffle(1 : Int(N/2))[1:Int(N/4)]
    r2 = shuffle(Int(N/2)+1 : N)[1:Int(N/4)]
    
    i3[r1] .= -1
    i3[r2] .= 1
    return [i1, i2, i3]
end

k = random_orthogonal_images(64)

println(dot(k[1],k[2]))
println(dot(k[1],k[3]))
println(dot(k[2],k[3]))