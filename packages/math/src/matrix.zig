pub fn Matrix(comptime T: type, comptime M: usize, comptime N: usize) type {
    return struct {
        elements: @Vector(T, M * N),

        // TODO
        // 矩阵相关函数
        // 列主序存储
        // 矩阵运算/数学函数等
        // 
    };
}
