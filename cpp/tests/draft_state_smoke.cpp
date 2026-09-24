#include "draft_state.hpp"
#include <cmath>
#include <cstring>
#include <limits>

template<class Label> void check(brook::Context &ctx,brook_dtype dtype) {
  constexpr int dense=257,foreground=199,reserve=1024,n=dense+reserve;
  auto labels=brook::allocate(ctx,{dense,1,1},dtype);
  auto dbf=brook::allocate(ctx,{dense,1,1},BROOK_F32),penalty=brook::allocate(ctx,{dense,1,1},BROOK_F32);
  std::vector<Label> original(dense);std::vector<float> d(dense),p(dense);std::vector<uint8_t> live(dense);
  for(int i=0;i<dense;++i) { original[i]=i<foreground?Label(i%7+1):0;d[i]=float(i)*0.125f;p[i]=float(i)*0.5f+1;live[i]=i<foreground; }
  BROOK_CUDA(cudaMemcpy(labels.data(),original.data(),original.size()*sizeof(Label),cudaMemcpyHostToDevice));
  BROOK_CUDA(cudaMemcpy(dbf.data(),d.data(),d.size()*4,cudaMemcpyHostToDevice));
  BROOK_CUDA(cudaMemcpy(penalty.data(),p.data(),p.size()*4,cudaMemcpyHostToDevice));
  brook::Buffer<uint8_t> alive(ctx,dense);alive.set(ctx,live.data(),live.size());
  brook::Buffer<unsigned long long> key(ctx,dense);key.clear(ctx);
  brook::Buffer<int> flags(ctx,dense);flags.clear(ctx);brook::Buffer<float> distance(ctx,dense);distance.clear(ctx);
  std::array<brook::Buffer<int>,7> lists;
  std::array<brook::Buffer<int>*,7> addresses;
  for(int i=0;i<7;++i) { lists[i]=brook::Buffer<int>(ctx,foreground,true);addresses[i]=&lists[i]; }
  auto old_labels=labels,old_dbf=dbf;
  brook::DraftVoxelBuffers state{labels,dbf,penalty,alive,key,flags,distance,addresses};
  try { brook::append_draft_reserve(ctx,state,dense,foreground,INT32_MAX);throw std::runtime_error("overflow was accepted"); }
  catch(const std::invalid_argument&) {}
  brook::append_draft_reserve(ctx,state,dense,foreground,reserve);
  std::vector<Label> new_labels(n),saved_labels(dense);labels.copy_to_host(new_labels.data(),new_labels.size()*sizeof(Label));
  old_labels.copy_to_host(saved_labels.data(),saved_labels.size()*sizeof(Label));
  if(saved_labels!=original || labels.data()==old_labels.data()) throw std::runtime_error("retained labels changed");
  std::vector<float> new_dbf(n),new_penalty(n),saved_dbf(dense);
  dbf.copy_to_host(new_dbf.data(),new_dbf.size()*4);penalty.copy_to_host(new_penalty.data(),new_penalty.size()*4);
  old_dbf.copy_to_host(saved_dbf.data(),saved_dbf.size()*4);
  if(saved_dbf!=d) throw std::runtime_error("retained DBF changed");
  auto a=alive.get(ctx,n);auto f=flags.get(ctx,n);auto k=key.get(ctx,n);auto dist=distance.get(ctx,n);
  for(int i=0;i<n;++i) {
    if(new_labels[i]!=(i<dense?original[i]:0) || new_dbf[i]!=(i<dense?d[i]:0) || new_penalty[i]!=(i<dense?p[i]:0) ||
        a[i]!=(i<dense?live[i]:0) || f[i]!=0 || k[i]!=~0ULL || dist[i]!=std::numeric_limits<float>::infinity())
      throw std::runtime_error("draft state prefix/tail mismatch");
  }
  for(int i=0;i<7;++i) {
    if(lists[i].size()!=foreground+reserve || !lists[i].storage()->managed) throw std::runtime_error("invalid grown worklist");
    for(int j=0;j<i;++j) if(lists[i].data()==lists[j].data()) throw std::runtime_error("worklist alias");
  }
}
int main() {
  brook::Context ctx(0);check<uint8_t>(ctx,BROOK_U8);check<uint16_t>(ctx,BROOK_U16);
  check<uint32_t>(ctx,BROOK_U32);check<uint64_t>(ctx,BROOK_U64);
}
